"""Package only verified runtime inputs/scripts; never copy large ONNX files."""
import hashlib
import json
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'bpu/deployment'
out.mkdir(exist_ok=True)
manifest={'models':[],'acceptance':'diagnostic prototype, not full acceptance'}
for scope,relative in [
    ('fusion','bpu/fusion_l128/a16_w8_sm8/embeddinggemma2_fusion_l128.bin'),
    ('vision','bpu/models/vision_a16_w8_sm8/embeddinggemma2_vision_s70.bin')]:
    state=json.loads((root/f'bpu/logs/multimodal_repair/{scope}_compile/state.json').read_text())
    assert state['status']=='complete' and not state['cpu_softmax_nodes']
    path=root/relative
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==state['runtime_sha256']
    manifest['models'].append({'scope':scope,'relative_path':relative,'sha256':digest,'bytes':path.stat().st_size})
manifest_path=out/'multimodal_deployment.json'
manifest_path.write_text(json.dumps(manifest,indent=2))
files=[root/f'scripts/{name}' for name in ['benchmark_multimodal_bpu.py','persistent_bpu.py','prepare_multimodal_inputs.py']]
files += [root/'bpu/fusion_l128/export_validation.json',root/'bpu/fusion_l128/reference_embeddings.npy',
          root/'bpu/fusion_l128/a16_w8_sm8/predictions.npy']
for folder in ['calibration','evaluation']:
    files+=list((root/'bpu/fusion_l128'/folder).rglob('*.bin'))
semantic=root/'bpu/results/multimodal_semantic_diagnosis'
files+=list(semantic.glob('*.json'))+list(semantic.glob('*.npz'))+list(semantic.glob('*.npy'))
archive=out/'multimodal-runtime-inputs.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in files:
        relative=path.relative_to(root).as_posix()
        assert path.is_file() and path.stat().st_size<8*1024*1024,relative
        tar.add(path,arcname=relative,recursive=False)
    tar.add(manifest_path,arcname='bpu/runtime/multimodal_deployment.json',recursive=False)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files)+1}))
