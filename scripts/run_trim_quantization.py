"""Finite, one-shot quantization runner; capture the actual subprocess exit."""
import json
import subprocess
import sys
import time
from pathlib import Path
root=Path('/work')
logs=root/'bpu/logs/multimodal_repair'
fresh='--fresh-builder' in sys.argv
prefix='vision_trim_fresh' if fresh else 'vision_trim'
state=logs/f'{prefix}_supervisor.json'
report={'status':'running','started_at':time.time(),'scope':'software quantization only'}
state.write_text(json.dumps(report,indent=2))
try:
    source=json.loads((root/'bpu/vision_trim576/results/vision_export_validation.json').read_text())
    assert source['status']=='complete' and source['trim_padding']
    with (logs/f'{prefix}_quantize.log').open('w') as out:
        command=['/usr/bin/python3.10',str(root/'scripts/quantize_trim_vision.py')]
        if fresh:command.append('--fresh-builder')
        completed=subprocess.run(command,
            cwd=root,stdout=out,stderr=subprocess.STDOUT,timeout=3600)
    (logs/f'{prefix}_quantize.exit').write_text(str(completed.returncode)+'\n')
    report.update(status='quantization_exited',exit_code=completed.returncode,finished_at=time.time())
except Exception as error:
    report.update(status='failed',error=repr(error),finished_at=time.time())
state.write_text(json.dumps(report,indent=2))
