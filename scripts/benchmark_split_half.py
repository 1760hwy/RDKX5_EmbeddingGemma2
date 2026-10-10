"""Board acceptance of one compiled half; not full multimodal acceptance."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from persistent_bpu import Runtime

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads(Path(args.manifest).read_text())
out = root / manifest['output']
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'summary.json').exists(), 'Refuse to overwrite experiment'
report = {'status': 'running', 'scope': 'first eight vision layers only',
          'full_multimodal_accepted': False, 'manifest': manifest,
          'cores': [], 'cases': [], 'started_at': time.time()}

def save():
    (out / 'summary.json').write_text(json.dumps(report, indent=2))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    a = np.asarray(values)
    return {'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'min_ms': float(a.min()),
            'max_ms': float(a.max()), 'count': len(values)}

runtime = None
try:
    save()
    data_path = root / manifest['data']
    assert digest(data_path) == manifest['data_sha256']
    data = np.load(data_path)
    for spec in manifest['models']:
        assert digest(root / spec['path']) == spec['sha256']
    current = data['pixels_0']
    # Core measurements keep only one chunk resident at a time.
    for i, spec in enumerate(manifest['models']):
        start = time.perf_counter()
        runtime = Runtime(root / spec['path'], [(1, 768, 1, 576)])
        load_seconds = time.perf_counter() - start
        assert runtime.output_shape == (1, 768, 1, 576)
        predicted, cold_ms = runtime.run([current])
        times, walls = [], []
        for _ in range(100):
            begin = time.perf_counter()
            repeated, ms = runtime.run([current])
            walls.append((time.perf_counter() - begin) * 1000)
            times.append(ms)
            assert np.array_equal(repeated, predicted), 'Non-deterministic output'
        close_begin = time.perf_counter()
        runtime.close()
        runtime = None
        row = {'chunk': i, 'load_seconds': load_seconds,
               'release_seconds': time.perf_counter() - close_begin,
               'cold_core_ms': cold_ms, 'core': stats(times),
               'adapter_wall': stats(walls), 'raw_core_ms': times,
               'raw_adapter_wall_ms': walls}
        report['cores'].append(row)
        np.save(out / f'chunk{i}_sample_output.npy', predicted)
        current = predicted
        save()
    # Six independent sequential passes count model loading, copies and release.
    for index, label in enumerate(manifest['cases']):
        current = data[f'pixels_{index}']
        begin = time.perf_counter()
        stages = []
        for i, spec in enumerate(manifest['models']):
            stage = time.perf_counter()
            runtime = Runtime(root / spec['path'], [(1, 768, 1, 576)])
            loaded = time.perf_counter()
            current, ms = runtime.run([current])
            runtime.close()
            runtime = None
            stages.append({'chunk': i, 'core_ms': ms,
                           'load_ms': (loaded - stage) * 1000,
                           'total_ms': (time.perf_counter() - stage) * 1000})
        total = (time.perf_counter() - begin) * 1000
        ref = data[f'reference_{index}']
        delta = current - ref
        relative = float(np.linalg.norm(delta) / np.linalg.norm(ref))
        cosine = float(np.dot(current.ravel(), ref.ravel()) /
                       (np.linalg.norm(current) * np.linalg.norm(ref)))
        np.save(out / f'{label}_boundary.npy', current)
        report['cases'].append({'case': label, 'cosine': cosine,
                                'max_abs_error': float(np.max(np.abs(delta))),
                                'relative_rmse': relative,
                                'sequential_half_ms': total, 'stages': stages})
        save()
    report.update(status='complete', finished_at=time.time(),
                  half_numeric_gate_passed=all(c['cosine'] >= .999 and
                                              c['relative_rmse'] < .01
                                              for c in report['cases']))
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
finally:
    if runtime is not None:
        runtime.close()
