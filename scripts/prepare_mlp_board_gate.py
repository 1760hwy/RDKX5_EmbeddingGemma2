"""Lock completed MLP candidate bins and unchanged FP32 references for board gate."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
folder=root/'bpu/vision_trim576/mlp_boundary_full16'
out=folder/'full_board_inputs'
assert not out.exists()
supervisor=json.loads((root/'bpu/logs/multimodal_repair/mlp_remaining_proto_verified_supervisor.json').read_text())
assert supervisor['status']=='complete'
assert (root/'bpu/logs/multimodal_repair/mlp_remaining_proto_verified_supervisor.exit').read_text().strip()=='0'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'bpu/vision_trim576/split2/full_board_inputs/manifest.json').read_text())
data_path=root/manifest['data'];assert sha(data_path)==manifest['data_sha256']
manifest['vision']=[]
for chunk in range(8):
    state=json.loads((folder/f'chunk{chunk}_compile/state.json').read_text())
    assert state['status']=='complete' and state['softmax_nodes']==14 and not state['cpu_softmax_nodes']
    assert state['target_nodes']==4 and not state['cpu_target_nodes']
    path=folder/f'chunk{chunk}_compile/embeddinggemma2_vision_mlp_boundary_chunk{chunk}.bin'
    assert sha(path)==state['runtime_sha256']
    manifest['vision'].append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path)))
arrays=dict(np.load(data_path))
for index,case in enumerate(manifest['images']):
    arrays[f'latent_{index}']=np.load(folder/f"{case['case']}_candidate_latent.npy")
out.mkdir()
np.savez(out/'data.npz',**arrays)
manifest.update(data=(out/'data.npz').relative_to(root).as_posix(),data_sha256=sha(out/'data.npz'),
                output='bpu/results/mlp_boundary_full_board_gate',
                quantized_sha256=json.loads((folder/'validation.json').read_text())['quantized_sha256'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('LOCKED',len(manifest['vision']),manifest['data_sha256'])
