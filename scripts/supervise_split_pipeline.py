"""Finite sequential continuation after the current board gate; no retries."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
logs = root / 'bpu/logs/multimodal_repair'
path = logs / 'split_pipeline_supervisor.json'
assert not path.exists(), 'Existing supervisor: inspect before restart'
report = {'status': 'waiting_gate', 'started_at': time.time(), 'jobs': []}

def save():
    path.write_text(json.dumps(report, indent=2))

try:
    save()
    deadline = time.monotonic() + 1800
    ready = logs / 'vision_split2_full_board.exit'
    while not ready.exists():
        assert time.monotonic() < deadline, 'Gate wait timed out'
        time.sleep(15)
    assert ready.read_text().strip() == '0', 'Board gate process failed'
    gate = json.loads((root / 'bpu/results/vision_split2_full_board_gate/summary.json').read_text())
    assert gate['status'] == 'complete' and gate['joint_board_gate_passed']
    assert len(gate['cores']) == 5 and all(c['core']['count'] == 100 for c in gate['cores'])
    for mode, repeats, limit in [('cpu', 3, 1800), ('bpu', 100, 7200)]:
        log = logs / f'split_pipeline_{mode}.log'
        exit_path = logs / f'split_pipeline_{mode}.exit'
        assert not log.exists() and not exit_path.exists(), 'Existing attempt; do not overwrite'
        report.update(status='running', active_mode=mode)
        save()
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, '-u', str(root/'scripts/benchmark_split_pipeline.py'),
                '--manifest', str(root/'bpu/vision_trim576/split2/full_board_inputs/pipeline_manifest.json'),
                '--mode', mode, '--repeats', str(repeats)], cwd=root, stdout=stream,
                stderr=subprocess.STDOUT, timeout=limit)
        exit_path.write_text(str(result.returncode) + '\n')
        report['jobs'].append({'mode': mode, 'exit_code': result.returncode})
        save()
        assert result.returncode == 0, f'{mode} benchmark failed; inspect log'
    report.update(status='complete', finished_at=time.time())
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
