"""One finite CPU/BPU batch acceptance sequence, without retries."""
import json
import subprocess
import time
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
logs = root / 'bpu/logs/multimodal_repair'
target = logs / 'batch_acceptance_supervisor.json'
assert not target.exists(), 'Inspect existing attempt rather than restart'
assert (logs / 'batch_resident_probe.exit').read_text().strip() == '0'
report = {'status': 'running', 'started_at': time.time(), 'jobs': []}


def save():
    target.write_text(json.dumps(report, indent=2))


try:
    save()
    for mode, repeats, images, deadline in [('cpu', 8, 18, 1800), ('bpu', 100, 100, 3600)]:
        result = root / ('bpu/results/vision_split2_batch_acceptance_' + mode) / 'summary.json'
        assert not result.exists(), 'Do not overwrite previous experiment'
        report['active_mode'] = mode
        save()
        command = [sys.executable, '-u', str(root / 'scripts/benchmark_batch_acceptance.py'),
                   '--manifest', str(root / 'bpu/vision_trim576/split2/full_board_inputs/pipeline_manifest.json'),
                   '--mode', mode, '--repeats', str(repeats), '--warm-images', str(images)]
        with (logs / ('batch_acceptance_' + mode + '.log')).open('w') as stream:
            job = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            report['child_pid'] = job.pid
            save()
            timed_out = False
            try:
                code = job.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                job.terminate()
                try:
                    job.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()
                code = 124
        (logs / ('batch_acceptance_' + mode + '.exit')).write_text(str(code) + '\n')
        report['jobs'].append({'mode': mode, 'exit_code': code, 'timed_out': timed_out})
        save()
        assert code == 0, (mode, code)
        summary = json.loads(result.read_text())
        assert summary['status'] == 'complete' and summary['all_sample_accuracy_passed']
        assert summary['all_image_retrieval_passed']
        assert all(summary[k]['warm_images'] == images for k in ['image', 'mixed'])
        if mode == 'bpu':
            assert summary['text']['count'] == 100 and summary['end_to_end_100_tested']
    report.update(status='complete', finished_at=time.time())
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
