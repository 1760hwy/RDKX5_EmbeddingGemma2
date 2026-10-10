"""Use original FP32/source references with the isolated repaired board candidate."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
folder=root/'bpu/vision_trim576/mlp_boundary_full16/full_board_inputs'
destination=folder/'pipeline_manifest.json'
assert not destination.exists()
manifest=json.loads((root/'bpu/vision_trim576/split2/full_board_inputs/pipeline_manifest.json').read_text())
manifest['gate_manifest']=json.loads((folder/'manifest.json').read_text())
manifest['cpu_baseline_note']='Same batch6 scheduling FP32 eager four threads, text8 and image/mixed18 warm; CPU is a small sample baseline, not CPU100.'
manifest['bpu_note']='New repaired bins: fusion resident; source preprocessing and every visual chunk load/free/copy counted in batch6 wall; each kind100 warm. Throughput amortization is not single-image response latency.'
destination.write_text(json.dumps(manifest,indent=2))
