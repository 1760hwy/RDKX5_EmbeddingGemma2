"""Export original vision layers for a fixed square-image patch grid.

This candidate accelerates the visual encoder, not the complete embedding
pipeline. Fixed positions/masks are precomputed from the official processor.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bpu'
trim_padding = '--trim-padding' in sys.argv
if trim_padding:
    # An isolated candidate: keep original graphs, calibration and reports.
    OUT = OUT / 'vision_trim576'
for directory in ['models', 'results', 'calibration/vision_pixels']:
    (OUT / directory).mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
model = SentenceTransformer(str(ROOT/'model'), device='cpu', local_files_only=True,
    config_kwargs={'audio_config': None},
    model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'})
model.eval()
auto = model[0].auto_model
vision = auto.vision_tower
names = ['GoldenGate.png', 'kitchen_painting.jpg']
features = [model[0].preprocess([{'image': Image.open(ROOT/'samples'/name).convert('RGB')}],
    processing_kwargs={'image': {'max_soft_tokens': 70}}) for name in names]
pos = features[0]['image_position_ids']
assert torch.equal(pos, features[1]['image_position_ids']), 'Different patch grids need separate export'
original_pos = pos
kept = ~(original_pos == -1).all(-1)[0]
if trim_padding:
    assert int(kept.sum()) == 576 and len(kept) == 630
    pos = original_pos[:, kept]
patches = pos.shape[1]
pad = (pos == -1).all(-1)
length = patches // 9
clamped = pos.clamp(min=0)
max_x = clamped[..., 0].max(-1, keepdim=True)[0] + 1
cells = torch.div(clamped, 3, rounding_mode='floor')
indices = cells[..., 0] + (max_x // 3) * cells[..., 1]
weights = torch.nn.functional.one_hot(indices.long(), length).float() / 9
valid = (weights != 0).any(1)
pool_weights = weights.transpose(1, 2)[:, valid[0], :] * (~pad).float()[:, None, :]

class VisionCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = vision.patch_embedder.input_proj
        self.layers = vision.encoder.layers
        self.embed = auto.embed_vision
        with torch.no_grad():
            positions = vision.patch_embedder._position_embeddings(pos, pad)
            cos, sin = vision.encoder.rotary_emb(torch.zeros(1, patches, vision.config.hidden_size), pos)
        self.register_buffer('positions', positions)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.register_buffer('mask', pad.float()[:, None, None, :] * -1e9)
        self.register_buffer('pool_weights', pool_weights)

    def forward(self, pixel_values):
        x = pixel_values.squeeze(2).transpose(1,2)
        h = self.input_proj(2 * (x - 0.5)) + self.positions
        for layer in self.layers:
            h = layer(h, attention_mask=self.mask, position_embeddings=(self.cos,self.sin), position_ids=pos)
        pooled = (self.pool_weights @ h) * vision.pooler.root_hidden_size
        projected = self.embed(inputs_embeds=pooled)
        return projected.transpose(1,2).unsqueeze(2)

wrapper = VisionCore().eval()
prepared, refs, checks = [], [], []
for i, f in enumerate(features):
    pixels = f['pixel_values'][:, kept] if trim_padding else f['pixel_values']
    x = pixels.transpose(1,2).unsqueeze(2)
    with torch.no_grad():
        reference = auto.get_image_features(f['pixel_values'], f['image_position_ids']).pooler_output[0]
        predicted = wrapper(x).squeeze(2).transpose(1,2)[0]
    error = float((predicted-reference).abs().max())
    cosine = float(torch.nn.functional.cosine_similarity(predicted.flatten(), reference.flatten(),dim=0))
    assert cosine > 0.99999 and error < 1e-4, (i,cosine,error)
    folder = OUT/'calibration/vision_pixels'
    folder.mkdir(parents=True, exist_ok=True)
    x.numpy().tofile(folder/f'{i:03d}.bin')
    prepared.append(x)
    refs.append(reference.numpy())
    checks.append({'sample': names[i], 'cosine':cosine,'max_abs_error':error})
    print('VISION_WRAPPER_VALIDATED', checks[-1], flush=True)
path = OUT/'models/vision_core_s70.onnx'
if '--validate-existing' not in sys.argv:
    torch.onnx.export(wrapper, prepared[0], str(path), opset_version=11,dynamo=False,
        input_names=['pixel_values'],output_names=['vision_features'],do_constant_folding=True)
graph = onnx.load(str(path))
onnx.checker.check_model(graph)
options=ort.SessionOptions()
options.intra_op_num_threads=4
options.inter_op_num_threads=1
session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
for i,x in enumerate(prepared):
    predicted=session.run(None,{'pixel_values':x.numpy()})[0].squeeze(2).transpose(0,2,1)[0]
    error=float(np.max(np.abs(predicted-refs[i])))
    cosine=float(np.dot(predicted.ravel(),refs[i].ravel())/(np.linalg.norm(predicted)*np.linalg.norm(refs[i])))
    relative_rmse=float(np.linalg.norm(predicted-refs[i])/np.linalg.norm(refs[i]))
    token_cosines=np.sum(predicted*refs[i],axis=1)/(np.linalg.norm(predicted,axis=1)*np.linalg.norm(refs[i],axis=1))
    print('VISION_ONNX_DIAGNOSTIC',i,{'max_abs':error,'cosine':cosine,'relative_rmse':relative_rmse,
        'min_token_cosine':float(token_cosines.min()),'reference_abs_max':float(np.abs(refs[i]).max())},flush=True)
    np.save(OUT/'results'/f'vision_reference_{i}.npy',refs[i])
    # These are unnormalized latent features (magnitude about 5), unlike unit
    # text embeddings. Judge scale-aware error and every token direction too.
    assert cosine>0.99999 and float(token_cosines.min())>0.99999 and relative_rmse<1e-4 and error<5e-4,(i,cosine,error,relative_rmse)
    checks[i].update(onnx_cosine=cosine,onnx_max_abs_error=error,
        onnx_relative_rmse=relative_rmse,onnx_min_token_cosine=float(token_cosines.min()))
np.save(OUT/'results/vision_reference_features.npy', np.asarray(refs))
report={'status':'complete','input_shape':list(prepared[0].shape),'output_shape':[1,512,1,refs[0].shape[0]],
    'trim_padding':trim_padding,'original_patch_count':int(original_pos.shape[1]),
    'valid_patch_count':int(kept.sum()),'weights_pruned':False,
    'max_soft_tokens':70,'fixed_grid':True,'checks':checks,'onnx_bytes':path.stat().st_size,
    'operators':dict(Counter(n.op_type for n in graph.graph.node)),
    'scope':'original visual encoder and projection; text network and final embedding are outside this graph'}
(OUT/'results/vision_export_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('VISION_ONNX_VALIDATED',report,flush=True)
