"""Bounded loading diagnosis after the complete baseline; no inference claims."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from persistent_bpu import Runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True,
                        choices=['fusion_only', 'fusion_chunk0', 'chunk0_chunk1'])
    parser.add_argument('--attempt', choices=['original', 'global_symbols'], default='original')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / 'bpu/results/vision_split2_pipeline_bpu/summary.json').read_text())
    exit_path = root / 'bpu/logs/multimodal_repair/split_pipeline_bpu.exit'
    assert baseline['status'] == 'complete' and exit_path.read_text().strip() == '0', \
        'Complete baseline must finish successfully before loading diagnosis'
    assert all(baseline[k]['count'] == 100 for k in ['text', 'image', 'mixed'])
    manifest = baseline['manifest']['gate_manifest']
    fusion = (manifest['fusion'], [(1, 512, 1, 128), (1, 1, 1, 128)])
    vision = [(spec, [(1, 768, 1, 576)]) for spec in manifest['vision'][:2]]
    selections = {'fusion_only': [fusion], 'fusion_chunk0': [fusion, vision[0]],
                  'chunk0_chunk1': vision}
    name = args.case if args.attempt == 'original' else args.case + '_' + args.attempt
    folder = root / 'bpu/results/residency_diagnosis' / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'summary.json'
    assert not target.exists(), 'Preserve the previous attempt; do not blindly retry'
    report = {'status': 'running', 'case': args.case, 'attempt': args.attempt, 'started_at': time.time(),
              'loads': [], 'inference_tested': False, 'co_residency_loaded': False,
              'note': 'Only model/input/output allocation, held together until cleanup. '
                      'Use observer logs for allocation peaks and final release evidence.'}
    runtimes = []

    def save():
        target.write_text(json.dumps(report, indent=2))

    save()
    try:
        for spec, shapes in selections[args.case]:
            path = root / spec['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
            begin = time.perf_counter()
            runtime = Runtime(path, shapes)
            runtimes.append(runtime)
            report['loads'].append({'model': spec, 'load_seconds': time.perf_counter() - begin,
                                    'output_shape': runtime.output_shape})
            save()
        report['co_residency_loaded'] = True
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        for runtime in reversed(runtimes):
            runtime.close()
        report['handles_closed'] = True
        report['finished_at'] = time.time()
        save()


if __name__ == '__main__':
    main()
