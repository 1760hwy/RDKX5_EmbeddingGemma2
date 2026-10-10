"""Separate vendor graph optimization / PTQ / board errors on fixed samples."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from horizon_nn.executor import ORTExecutor

parser = argparse.ArgumentParser()
parser.add_argument("scope", choices=["text", "vision"])
args = parser.parse_args()
scope = args.scope
root = Path("/work/bpu")
prefix = "embeddinggemma2_" + ("text_l32" if scope == "text" else "vision_s70")
models = root / "models" / ("compiled_text_l32_bounded" if scope == "text" else "compiled_vision_s70_bounded")
refs = np.load(root / "results" / ("reference_embeddings.npy" if scope == "text" else "vision_reference_features.npy"))
board = np.load(root / "results" / f"{scope}_runtime" / "predictions.npy")
report = {"scope": scope, "status": "running", "started_at": time.time(), "stages": {}}
destination = root / "results" / f"{scope}_quantization_diagnosis.json"

def persist():
    destination.write_text(json.dumps(report, indent=2))

def cosine(a, b):
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))

persist()
try:
    for stage in ["optimized_float", "quantized"]:
        executor = ORTExecutor(str(models / f"{prefix}_{stage}_model.onnx")).to("cpu").create_session()
        checks = []
        predictions = []
        for i in range(2):
            if scope == "text":
                inputs = {
                    "inputs_embeds": np.fromfile(root / f"calibration/inputs_embeds/{i:03d}.bin", np.float32).reshape(1, 512, 1, 32),
                    "attention_mask": np.fromfile(root / f"calibration/attention_mask/{i:03d}.bin", np.float32).reshape(1, 1, 1, 32),
                }
            else:
                inputs = {"pixel_values": np.fromfile(root / f"calibration/vision_pixels/{i:03d}.bin", np.float32).reshape(1, 768, 1, 630)}
            started = time.time()
            outputs = executor.forward(inputs)
            assert len(outputs) == 1, list(outputs)
            output = next(iter(outputs.values()))
            assert output.shape == ((1, 768, 1, 1) if scope == "text" else (1, 512, 1, 64)), output.shape
            result = output.reshape(768) if scope == "text" else output.squeeze(2).transpose(0, 2, 1)[0]
            assert np.isfinite(result).all() and np.linalg.norm(result) > 0
            checks.append({"case": i, "reference_cosine": cosine(result, refs[i]), "board_cosine": cosine(result, board[i]),
                           "max_reference_error": float(np.max(np.abs(result - refs[i]))),
                           "max_board_error": float(np.max(np.abs(result - board[i]))), "seconds": time.time() - started})
            predictions.append(result)
            print(stage, checks[-1], flush=True)
        np.save(root / "results" / f"{scope}_{stage}_diagnostic_predictions.npy", np.asarray(predictions))
        report["stages"][stage] = checks
        persist()
        del executor
        gc.collect()
    report.update(status="complete", finished_at=time.time())
    persist()
except Exception as error:
    report.update(status="failed", error=str(error), finished_at=time.time())
    persist()
    raise
