"""Compare selected activation tensors with calibration ON versus OFF.

This localizes fake-quantization error; it is not a bit-exact compiled test.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from horizon_nn.common import add_model_output
from horizon_nn.ir import OnnxModel
from horizon_nn.executor import ORTExecutor

parser=argparse.ArgumentParser()
parser.add_argument("scope",choices=["text","vision"])
args=parser.parse_args()
scope=args.scope
root=Path("/work/bpu")
prefix="embeddinggemma2_"+("text_l32" if scope=="text" else "vision_s70")
folder="compiled_text_l32_bounded" if scope=="text" else "compiled_vision_s70_bounded"
source=root/"models"/folder/f"{prefix}_calibrated_model.onnx"
layerprefix="/network/layers." if scope=="text" else "/layers."
inputs=({"inputs_embeds":np.fromfile(root/"calibration/inputs_embeds/000.bin",np.float32).reshape(1,512,1,32),
         "attention_mask":np.fromfile(root/"calibration/attention_mask/000.bin",np.float32).reshape(1,1,1,32)}
        if scope=="text" else {"pixel_values":np.fromfile(root/"calibration/vision_pixels/000.bin",np.float32).reshape(1,768,1,630)})
report={"scope":scope,"status":"running","note":"Single sample fake-quant activation trace; sensitivity evidence, not board proof","checks":[]}
destination=root/"results"/f"{scope}_calibration_trace.json"
def save(): destination.write_text(json.dumps(report,indent=2))
save()
try:
    model=OnnxModel(str(source))
    selected=[]
    for node in model.graph.nodes:
        first=node.name.startswith(layerprefix+"0/") and node.op_type in ("Conv","MatMul","Pow","ReduceMean")
        residual=node.name.startswith(layerprefix) and node.name.split("/")[-1]=="Add_1" and "/self_attn/" not in node.name and "/mlp/" not in node.name
        softmax=node.name.startswith(layerprefix+"0/") and "FROM_QUANTIZED_SOFTMAX" in node.name
        if first or residual or softmax:
            selected.append((node.name,node.op_type,node.output_names[0]))
    assert selected
    model=add_model_output(model,output_tensors=[x[2] for x in selected])
    # Keep only a manageable set of diagnostic output tensors.
    executor=ORTExecutor(model).to("cpu").create_session()
    quant=executor.forward(inputs,output_names=[x[2] for x in selected])
    del executor
    gc.collect()
    for node in model.graph.type2nodes["HzCalibration"]: node.switch="OFF"
    executor=ORTExecutor(model).to("cpu").create_session()
    floating=executor.forward(inputs,output_names=[x[2] for x in selected])
    for name,op,tensor in selected:
        a,b=quant[tensor].ravel(),floating[tensor].ravel()
        assert a.shape==b.shape
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        denom=float(np.linalg.norm(a)*np.linalg.norm(b)) if finite else 0
        row={"node":name,"op":op,"shape":list(quant[tensor].shape),"cosine":float(a@b/denom) if denom else None,
             "relative_rmse":float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12)) if finite else None,
             "float_absmax":float(np.max(np.abs(b[np.isfinite(b)]))) if np.isfinite(b).any() else None,
             "quant_absmax":float(np.max(np.abs(a[np.isfinite(a)]))) if np.isfinite(a).any() else None,
             "float_nonfinite_fraction":float(np.mean(~np.isfinite(b))),"quant_nonfinite_fraction":float(np.mean(~np.isfinite(a))),
             "float_zero_fraction":float(np.mean(b==0)),"quant_zero_fraction":float(np.mean(a==0))}
        report["checks"].append(row)
        save()
        print(row,flush=True)
    report["status"]="complete"
    save()
except Exception as error:
    report.update(status="failed",error=str(error))
    save()
    raise
