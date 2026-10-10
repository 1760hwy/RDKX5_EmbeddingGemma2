"""Isolated new chunk: six numerical boundaries, core100, official profile3.

Includes model load and adapter wall separately. This is not whole-model
semantic, throughput or end-to-end acceptance.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from persistent_bpu import Runtime

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'bpu/vision_trim576/mlp_boundary_full16/chunk0_compile'
manifest = json.loads((FOLDER/'board_probe_manifest.json').read_text())
OUT = ROOT/manifest['output']
OUT.mkdir(exist_ok=False)
report = dict(status='running',started_at=time.time(),phase='hashes',manifest=manifest,
              full_multimodal_accepted=False,end_to_end_100_tested=False,checks=[])


def save():
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


runtime = None
try:
    save()
    model = ROOT/manifest['model']
    data_path = FOLDER/'board_probe_data.npz'
    assert sha(model)==manifest['model_sha256'] and sha(data_path)==manifest['data_sha256']
    data = np.load(data_path)
    report['phase'] = 'load'
    save()
    start = time.perf_counter()
    runtime = Runtime(model,[(1,768,1,576)])
    report['load_ms'] = (time.perf_counter()-start)*1000
    assert runtime.output_shape == (1,768,1,576)
    report['phase'] = 'accuracy'
    save()
    for index, label in enumerate(manifest['cases']):
        start = time.perf_counter()
        y, core_ms = runtime.run([data[f'pixels_{index}']])
        wall_ms = (time.perf_counter()-start)*1000
        reference = data[f'reference_{index}']
        error = float(np.max(np.abs(y-reference)))
        relative = float(np.linalg.norm(y-reference)/np.linalg.norm(reference))
        row = dict(case=label,max_abs=error,relative_rmse=relative,core_ms=core_ms,adapter_wall_ms=wall_ms)
        report['checks'].append(row)
        np.save(OUT/f'{label}_boundary.npy',y)
        print(row,flush=True)
        save()
        assert error < 1e-4 and relative < 1e-5, (label,error,relative)
    report['accuracy_passed'] = True
    report['phase'] = 'core100'
    save()
    core, wall = [], []
    reference = np.load(OUT/f'{manifest["cases"][0]}_boundary.npy')
    for index in range(100):
        start = time.perf_counter()
        y, ms = runtime.run([data['pixels_0']])
        wall.append((time.perf_counter()-start)*1000)
        core.append(ms)
        assert np.array_equal(y,reference), 'Nondeterministic repeated output'
    report['core100'] = dict(count=100,raw_core_ms=core,raw_adapter_wall_ms=wall,
                            mean_core_ms=float(np.mean(core)),p95_core_ms=float(np.percentile(core,95)),
                            mean_adapter_wall_ms=float(np.mean(wall)),deterministic=True)
    runtime.close()
    runtime = None
    report['phase'] = 'profile3'
    save()
    profile = OUT/'profile'
    profile.mkdir()
    inp = OUT/'valid_input.bin'
    np.ascontiguousarray(data['pixels_0'],dtype=np.float32).tofile(inp)
    command = ['hrt_model_exec','perf','--model_file',str(model),'--input_file',str(inp),
               '--frame_count','3','--thread_num','1','--profile_path',str(profile)]
    with (OUT/'official_profile_command.log').open('w') as log:
        completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    report['profile_exit'] = completed.returncode
    assert completed.returncode==0
    report.update(status='complete',finished_at=time.time(),
                  note='Only new first-two-layer chunk; full new vision/fusion board acceptance remains untested.')
    save()
except Exception as error:
    report.update(status='failed',error=repr(error),finished_at=time.time())
    save()
    raise
finally:
    if runtime is not None:
        runtime.close()
