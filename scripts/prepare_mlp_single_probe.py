"""Isolate a small batch-one probe from the already verified acceptance runner."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/benchmark_mlp_batch_acceptance.py').read_text()
changes = {
    "bpu/results/mlp_batch_acceptance_": "bpu/results/mlp_single_probe_",
    "for repeat in range(-1, (args.warm_images+5)//6):": "for repeat in range(-1, args.warm_images):",
    "size = 6 if repeat == -1 else min(6, args.warm_images-repeat*6)": "size = 1",
    "report['batch_size'] = 6": "report['batch_size'] = 1",
    "batch6": "batch1",
    "six-image batch": "single-image request",
}
for old, new in changes.items():
    assert old in source, old
    source = source.replace(old, new)
target = root / 'scripts/benchmark_mlp_single_probe.py'
assert not target.exists()
target.write_text(source)
supervisor = (root / 'scripts/supervise_mlp_batch_acceptance.py').read_text()
supervisor = supervisor.replace('mlp_batch_acceptance_', 'mlp_single_probe_')
supervisor = supervisor.replace('benchmark_mlp_batch_acceptance.py', 'benchmark_mlp_single_probe.py')
supervisor = supervisor.replace("[('cpu', 8, 18, 1800), ('bpu', 100, 100, 3600)]", "[('cpu', 8, 3, 600), ('bpu', 8, 3, 600)]")
supervisor = supervisor.replace("        if mode == 'bpu':\n            assert summary['text']['count'] == 100 and summary['end_to_end_100_tested']", "        assert summary['text']['count'] == 8\n        assert not summary['end_to_end_100_tested']")
target = root / 'scripts/supervise_mlp_single_probe.py'
assert not target.exists()
target.write_text(supervisor)
