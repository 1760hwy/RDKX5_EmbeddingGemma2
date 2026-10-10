"""Add isolated CPU text-core timestamps missing from the original wall runner."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
p = (root / 'scripts/benchmark_mlp_batch_acceptance.py').read_text()
p = p.replace('bpu/results/mlp_batch_acceptance_', 'bpu/results/mlp_text_core_probe_')
old = """        if args.mode == 'cpu':
            with torch.no_grad():
                pred = cpu_fusion(auto, arrays)
        else:"""
new = """        if args.mode == 'cpu':
            core_start = time.perf_counter()
            with torch.no_grad():
                pred = cpu_fusion(auto, arrays)
            stages.append({'chunk': 'fusion_cpu', 'core_ms': (time.perf_counter()-core_start)*1000})
        else:"""
assert old in p
p = p.replace(old, new)
p = p.replace("    for kind in ['image', 'mixed']:", '    for kind in []:')
p = p.replace('full_acceptance_passed=False)', "full_acceptance_passed=False, diagnostic_scope='CPU text core only, eight warm captions')")
target = root / 'scripts/benchmark_mlp_text_core_probe.py'
assert not target.exists()
target.write_text(p)
