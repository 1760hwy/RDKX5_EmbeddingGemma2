"""Verify adapter outputs against previously pulled official runtime outputs."""
import json
from pathlib import Path
import numpy as np
from persistent_bpu import Runtime
root=Path(__file__).resolve().parents[1]
out=root/'bpu/results/persistent_runtime_validation.json'
report={'status':'running','cases':[]}
try:
    for scope,shapes in [('text',[(1,512,1,32),(1,1,1,32)]),('vision',[(1,768,1,630)])]:
        name='embeddinggemma2_'+('text_l32' if scope=='text' else 'vision_s70')
        runtime=Runtime(root/f'bpu/models/{name}.bin',shapes)
        try:
            folders=['inputs_embeds','attention_mask'] if scope=='text' else ['vision_pixels']
            arrays=[np.fromfile(root/f'bpu/calibration/{folder}/000.bin',np.float32).reshape(shape) for folder,shape in zip(folders,shapes)]
            expected=np.load(root/f'bpu/results/{scope}_runtime/predictions.npy')[0]
            for repeat in range(2):
                raw,ms=runtime.run(arrays)
                actual=raw.reshape(768) if scope=='text' else raw.squeeze(2).transpose(0,2,1)[0]
                error=float(np.max(np.abs(actual-expected)))
                assert error<1e-6,(scope,error)
                report['cases'].append({'scope':scope,'repeat':repeat,'max_abs_error':error,'infer_wait_ms':ms})
        finally:
            runtime.close()
    report['status']='complete'
except Exception as error:
    report.update(status='failed',error=repr(error));raise
finally:
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
