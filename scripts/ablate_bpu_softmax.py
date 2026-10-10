"""Software-only sensitivity experiments; not deployable compiled models.

Retain INT8 attention logits, vary probability/intermediate calibration only.
Float ablations diagnose errors; they do not satisfy the requested BPU goal.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from horizon_nn.ir import OnnxModel
from horizon_nn.executor import ORTExecutor

parser = argparse.ArgumentParser()
parser.add_argument("scope", choices=["text", "vision"])
parser.add_argument("--modes",default="all_calibration_off,baseline,softmax_probabilities_float,softmax_probabilities_int16")
parser.add_argument("--experiment",default="softmax")
args = parser.parse_args()
scope = args.scope
root = Path("/work/bpu")
prefix = "embeddinggemma2_" + ("text_l32" if scope == "text" else "vision_s70")
folder = "compiled_text_l32_bounded" if scope == "text" else "compiled_vision_s70_bounded"
source = root / "models" / folder / f"{prefix}_calibrated_model.onnx"
refs = np.load(root / "results" / ("reference_embeddings.npy" if scope == "text" else "vision_reference_features.npy"))
inputs = ({"inputs_embeds": np.fromfile(root / "calibration/inputs_embeds/000.bin", np.float32).reshape(1,512,1,32),
           "attention_mask": np.fromfile(root / "calibration/attention_mask/000.bin", np.float32).reshape(1,1,1,32)}
          if scope == "text" else {"pixel_values": np.fromfile(root / "calibration/vision_pixels/000.bin", np.float32).reshape(1,768,1,630)})
assert args.experiment in ("softmax","normalization","weight_feature","precision_sweep","range_sweep","softmax_range")
destination = root / "results" / f"{scope}_{args.experiment}_ablation.json"
report = {"status":"running", "scope":scope, "note":"One calibration sample, software fake quantization only; no board/compilation claim", "experiments":[]}
def persist():
    destination.write_text(json.dumps(report,indent=2))
persist()
try:
    modes=args.modes.split(',')
    assert all(m in ("all_calibration_off","baseline","softmax_probabilities_float","softmax_probabilities_int16","normalization_float","normalization_int16","weight_float","feature_float","feature_float_keep_softmax_int8","feature_int16_keep_softmax_int8","feature_int16","feature_int8_clip_half","feature_int8_clip_quarter","feature_int16_norm_float_sm8","feature_int16_sm8_logits_half","feature_int16_sm8_logits_threequarters") for m in modes)
    for mode in modes:
        model = OnnxModel(str(source))
        changed = []
        for node in model.graph.type2nodes["HzCalibration"]:
            # Softmax decomposition's sub/exp/reducesum/reciprocal/output;
            # excludes input logits, which retain original INT8 calibration.
            probability = "Softmax_output_0" in node.name
            normalization="norm/" in node.name
            softmax_boundary="Softmax" in node.name or any("FROM_QUANTIZED_SOFTMAX" in consumer.name for consumer in node.next_ops)
            if mode in ("feature_int16_sm8_logits_half","feature_int16_sm8_logits_threequarters") and node.tensor_type=="feature":
                if not softmax_boundary:
                    node.qtype="int16"
                    changed.append(node.name)
                elif "Softmax" not in node.name and node.thresholds:
                    factor=0.5 if mode=="feature_int16_sm8_logits_half" else 0.75
                    node.thresholds=[float(t)*factor for t in node.thresholds]
                    changed.append(node.name)
            elif mode == "feature_int16_norm_float_sm8" and node.tensor_type=="feature" and not softmax_boundary:
                if normalization:
                    node.switch="OFF"
                else:
                    node.qtype="int16"
                changed.append(node.name)
            elif mode in ("feature_int8_clip_half","feature_int8_clip_quarter") and node.tensor_type=="feature" and not softmax_boundary and node.thresholds:
                factor=0.5 if mode=="feature_int8_clip_half" else 0.25
                node.thresholds=[float(t)*factor for t in node.thresholds]
                changed.append(node.name)
            elif (mode == "all_calibration_off" or (mode == "softmax_probabilities_float" and probability)
                or (mode == "normalization_float" and normalization)
                or (mode == "weight_float" and node.tensor_type=="weight")
                or (mode == "feature_float" and node.tensor_type=="feature")
                or (mode == "feature_float_keep_softmax_int8" and node.tensor_type=="feature" and not softmax_boundary)):
                node.switch = "OFF"
                changed.append(node.name)
            elif (mode == "softmax_probabilities_int16" and probability) or (mode == "normalization_int16" and normalization):
                node.qtype = "int16"
                changed.append(node.name)
            elif node.tensor_type=="feature" and (mode=="feature_int16" or (mode=="feature_int16_keep_softmax_int8" and not softmax_boundary)):
                node.qtype="int16"
                changed.append(node.name)
        executor = ORTExecutor(model).to("cpu").create_session()
        started = time.time()
        outputs = executor.forward(inputs)
        raw = next(iter(outputs.values()))
        assert raw.shape == ((1,768,1,1) if scope == "text" else (1,512,1,64)), raw.shape
        result = raw.reshape(768) if scope == "text" else raw.squeeze(2).transpose(0,2,1)[0]
        ref = refs[0]
        assert np.isfinite(result).all() and np.linalg.norm(result)>0
        row = {"mode":mode, "case":0, "changed_count":len(changed), "changed_nodes":changed,
               "reference_cosine":float(result.ravel()@ref.ravel()/(np.linalg.norm(result)*np.linalg.norm(ref))),
               "relative_rmse":float(np.linalg.norm(result-ref)/np.linalg.norm(ref)), "seconds":time.time()-started}
        report["experiments"].append(row)
        persist()
        print({k:v for k,v in row.items() if k!="changed_nodes"},flush=True)
        del executor, model
        gc.collect()
    report["status"]="complete"
    persist()
except Exception as error:
    report.update(status="failed",error=str(error))
    persist()
    raise
