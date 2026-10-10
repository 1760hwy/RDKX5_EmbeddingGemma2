"""Resume vendor compilation using byte-identical HBIR and identical flags.

No vendor installation is modified. Preserve the timed-out conversion logs.
The saved quantized model is repacked through the vendor runtime converter;
numerical acceptance is performed separately on the physical board.
"""
import ast
import hashlib
import importlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path

scope = sys.argv[1]
assert scope in ("text", "vision")
root = Path("/work/bpu")
config = root / ("text_core_l32.yaml" if scope == "text" else "vision_core_s70.yaml")
output = root / "models" / f"compiled_{'text_l32' if scope == 'text' else 'vision_s70'}_bounded"
name = f"embeddinggemma2_{'text_l32' if scope == 'text' else 'vision_s70'}"
cache = root / "cache" / f"{scope}_timeout"
logdir = root / "logs" / f"{scope}_recovered"
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
assert json.loads((root / "results" / f"{scope}_bounded_mask_validation.json").read_text())["status"] == "complete"

def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

flags = {}
current = None
for line in (root / "logs" / f"{scope}_configured_bounded" / "compile.log").read_text().splitlines():
    match = re.search(r"Compile submodel: (\S+)", line)
    if match:
        current = match.group(1)
    if "hbdk-cc parameters:" in line and current:
        flags[current] = ast.literal_eval(line.split("hbdk-cc parameters:", 1)[1])

hb = importlib.import_module("horizon_nn.compiler.hybrid_build")
mapper = importlib.import_module("horizon_tc_ui.hb_mapper_makertbin")
from horizon_nn.ir import OnnxModel
from horizon_nn.common import HbdkDictParser, infer_shapes, convert_reshape_target_shape_to_positive
import onnx

runner = mapper.MakertbinRunner(str(config), "onnx")
runner._parse_conf()
params = runner._get_build_hbdk_params()
quantized = output / f"{name}_quantized_model.onnx"
prepared = infer_shapes(OnnxModel(str(quantized)), OnnxModel(str(output / f"{name}_original_float_model.onnx")), input_shape={})
prepared = convert_reshape_target_shape_to_positive(prepared)
builder = hb.HybridBuilder(prepared, "bayes-e", HbdkDictParser(params), output_path=str(output))
state = {"scope": scope, "quantized_sha256": digest(quantized), "subgraphs": builder.num_submodel(), "reused": [], "compiled": [], "status": "running"}

def save_state():
    (logdir / "recovery.json").write_text(json.dumps(state, indent=2))

save_state()
logging.info("Recovery has %s total subgraphs", state["subgraphs"])
original_cc = hb.hbdk_cc

def resume_cc(hbir_model, hbm_model, march, params=None, dump_all_models=False):
    subname = Path(hbir_model).stem
    old_ir, old_bin = cache / f"{subname}.hbir", cache / f"{subname}.hbm"
    assert march == "bayes-e"
    if old_ir.exists() and old_bin.exists() and old_bin.stat().st_size:
        assert digest(old_ir) == digest(hbir_model), f"HBIR mismatch: {subname}"
        assert flags[subname] == params, f"Compiler flags mismatch: {subname}"
        shutil.copy2(old_bin, hbm_model)
        state["reused"].append({"name": subname, "hbir_sha256": digest(old_ir), "hbm_sha256": digest(old_bin), "flags": params})
        logging.info("Verified reuse: %s", subname)
    else:
        original_cc(hbir_model, hbm_model, march, params, dump_all_models)
        shutil.copy2(hbir_model, cache / f"{subname}.hbir")
        shutil.copy2(hbm_model, cache / f"{subname}.hbm")
        state["compiled"].append(subname)
        logging.info("Persisted newly compiled subgraph: %s", subname)
    save_state()

hb.hbdk_cc = resume_cc
hybrid = builder.hybrid_model()
onnx.save(hybrid, str(output / f"{name}_recovered_hybrid_model.onnx"))
builder.model_info(str(logdir / "node_placement.log"))
logging.info("Saved recovered hybrid model; runtime packaging follows")

def use_verified_hybrid(onnx_file=None, **kwargs):
    assert Path(onnx_file).resolve() == Path(runner.mp_conf.onnx_model).resolve()
    return hybrid

mapper.build_onnx = use_verified_hybrid
runner.run("1.24.3")
runtime = output / f"{name}.bin"
assert runtime.exists() and runtime.stat().st_size > 0
state.update(status="complete", runtime_sha256=digest(runtime), runtime_bytes=runtime.stat().st_size)
save_state()
