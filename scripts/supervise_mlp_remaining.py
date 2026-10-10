"""Bounded sequential preparation and compilation, with no retries."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/work/bpu/vision_trim576/mlp_boundary_full16')
LOGS = Path('/work/bpu/logs/multimodal_repair')
REPORT = LOGS/'mlp_remaining_proto_verified_supervisor.json'


def main():
    assert not REPORT.exists(), 'Do not duplicate or overwrite a previous run'
    state = dict(status='running', started_at=time.time(), jobs=[], child_pid=None)
    def save():
        REPORT.write_text(json.dumps(state, indent=2))
    def run(name, arguments, limit):
        log = LOGS/f'{name}.log'
        code = LOGS/f'{name}.exit'
        assert not log.exists() and not code.exists(), name
        with log.open('w') as output:
            child = subprocess.Popen([sys.executable, '-u', *arguments], stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(active_job=name, child_pid=child.pid)
            save()
            try:
                result = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                result = 124
        code.write_text(str(result)+'\n')
        state['jobs'].append(dict(name=name, exit_code=result))
        state['child_pid'] = None
        save()
        assert result == 0, (name, result)
    try:
        save()
        run('mlp_remaining_proto_verified_prepare', ['/work/scripts/prepare_mlp_remaining_chunks.py'], 1800)
        gate = json.loads((ROOT/'split2_proto_verified/validation.json').read_text())
        assert gate['status']=='complete' and gate['software_equivalence_passed']
        for chunk in range(1, 8):
            run(f'mlp_boundary_chunk{chunk}_compile', ['/work/scripts/compile_mlp_boundary_chunk0.py', '--chunk', str(chunk)], 7200)
            result = json.loads((ROOT/f'chunk{chunk}_compile/state.json').read_text())
            assert result['status']=='complete' and result['software_slice_equivalence_passed']
            assert result['softmax_nodes']==14 and not result['cpu_softmax_nodes']
            assert result['target_nodes']==4 and not result['cpu_target_nodes']
            assert hashlib.sha256(Path(result['runtime']).read_bytes()).hexdigest()==result['runtime_sha256']
        state.update(status='complete', active_job=None, finished_at=time.time())
        save()
    except Exception as error:
        state.update(status='failed', error=repr(error), finished_at=time.time())
        save()
        raise


if __name__=='__main__':
    main()
