"""One-shot gated compile with a recorded real exit and a four-hour bound."""
import json,subprocess,time
from pathlib import Path
root=Path('/work')
logs=root/'bpu/logs/multimodal_repair'
state=logs/'vision_trim_compile_supervisor.json'
report={'status':'running','started_at':time.time(),'scope':'vision_trim compilation only'}
state.write_text(json.dumps(report,indent=2))
try:
    gate=json.loads((root/'bpu/vision_trim576/a16_w8_sm8_fresh/validation.json').read_text())
    assert gate['status']=='complete' and gate['joint_gate_passed']
    assert (logs/'vision_trim_fresh_quantize.exit').read_text().strip()=='0'
    with (logs/'vision_trim_compile.log').open('w') as out:
        result=subprocess.run(['/usr/bin/python3.10',str(root/'scripts/compile_multimodal_candidate.py'),'vision_trim'],
            cwd=root,stdout=out,stderr=subprocess.STDOUT,timeout=14400)
    (logs/'vision_trim_compile.exit').write_text(str(result.returncode)+'\n')
    report.update(status='compile_exited',exit_code=result.returncode,finished_at=time.time())
except Exception as error:
    report.update(status='failed',error=repr(error),finished_at=time.time())
state.write_text(json.dumps(report,indent=2))
