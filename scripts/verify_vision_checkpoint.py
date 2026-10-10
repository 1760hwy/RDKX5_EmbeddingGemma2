"""Bind the legacy vision report to an unchanged graph by replaying outputs."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from horizon_nn.ir import OnnxModel
from horizon_nn.executor import ORTExecutor

root=Path('/work/bpu')
folder=root/'models/vision_a16_w8_sm8'
path=folder/'embeddinggemma2_vision_s70_quantized_model.onnx'
digest=hashlib.sha256(path.read_bytes()).hexdigest()
expected=np.load(folder/'calibration_predictions.npy')
executor=ORTExecutor(OnnxModel(str(path))).to('cpu').create_session()
checks=[]
for i,ref in enumerate(expected):
    pixels=np.fromfile(root/f'calibration/vision_pixels/{i:03d}.bin',np.float32).reshape(1,768,1,630)
    raw=next(iter(executor.forward({'pixel_values':pixels}).values()))
    actual=raw.squeeze(2).transpose(0,2,1)[0]
    check={'case':i,'max_abs_error':float(np.max(np.abs(actual-ref))),'exact_equal':bool(np.array_equal(actual,ref))}
    checks.append(check);print(check,flush=True)
    assert check['exact_equal'], 'Saved semantic features cannot be bound to this graph'
assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Graph changed during verification'
report_path=root/'results/vision_mixed_precision_validation.json'
report=json.loads(report_path.read_text())
assert report['status']=='complete' and not report['numerical_gate_passed']
report['quantized_sha256']=digest
report['provenance_replay']={'verified_at':time.time(),'checks':checks,
    'note':'Missing legacy graph hash recovered by exact replay; original intermediate accuracy failure retained'}
report_path.write_text(json.dumps(report,indent=2))
print('VERIFIED_SHA256',digest,flush=True)
