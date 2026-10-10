"""Same 576-patch/128-token CPU and sequential BPU pipeline benchmarks."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
from persistent_bpu import Runtime
from prepare_multimodal_inputs import prepare, fuse
from benchmark import state

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--mode', choices=['cpu', 'bpu'], required=True)
parser.add_argument('--repeats', type=int, required=True)
args = parser.parse_args()
assert args.repeats > 0
root = Path(__file__).resolve().parents[1]
manifest = json.loads(Path(args.manifest).read_text())
out = root / ('bpu/results/vision_split2_pipeline_' + args.mode)
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'summary.json').exists(), 'Refuse to overwrite benchmark'
report = {'status': 'running', 'mode': args.mode, 'started_at': time.time(),
    'repeats_per_mode': args.repeats, 'vision_patches': 576, 'fusion_tokens': 128,
    'threads': 4, 'manifest': manifest, 'samples': [], 'state_before': state(),
    'note': 'Times include source image read/letterbox/processor/lookup, copies, inference and normalization. BPU chunk/fusion load and release counted every call; initial CPU model load separate.'}

def save():
    (out / 'summary.json').write_text(json.dumps(report, indent=2))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    a = np.asarray(values)
    return {'count': len(values), 'mean_ms': float(a.mean()),
        'median_ms': float(np.median(a)), 'p95_ms': float(np.percentile(a, 95)),
        'min_ms': float(a.min()), 'max_ms': float(a.max())}

class FixedVision(torch.nn.Module):
    """Exact FP32 trim wrapper from export_bpu_vision.py --trim-padding."""
    def __init__(self, auto, positions):
        super().__init__()
        vision = auto.vision_tower
        self.input_proj, self.layers, self.embed = (
            vision.patch_embedder.input_proj, vision.encoder.layers, auto.embed_vision)
        self.pos = positions
        pad = (positions == -1).all(-1)
        cells = torch.div(positions.clamp(min=0), 3, rounding_mode='floor')
        max_x = positions.clamp(min=0)[..., 0].max(-1, keepdim=True)[0] + 1
        indices = cells[..., 0] + (max_x // 3) * cells[..., 1]
        weights = torch.nn.functional.one_hot(indices.long(), 64).float() / 9
        valid = (weights != 0).any(1)
        self.pool = weights.transpose(1, 2)[:, valid[0], :] * (~pad).float()[:, None, :]
        self.positions = vision.patch_embedder._position_embeddings(positions, pad)
        self.cos, self.sin = vision.encoder.rotary_emb(torch.zeros(1, 576, vision.config.hidden_size), positions)
        self.mask = pad.float()[:, None, None, :] * -1e9
        self.scale = vision.pooler.root_hidden_size

    def forward(self, pixels):
        x = pixels.squeeze(2).transpose(1, 2)
        hidden = self.input_proj(2 * (x - .5)) + self.positions
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=self.mask,
                position_embeddings=(self.cos, self.sin), position_ids=self.pos)
        return self.embed(inputs_embeds=(self.pool @ hidden) * self.scale).transpose(1, 2).unsqueeze(2)

def cpu_fusion(auto, arrays):
    x = torch.from_numpy(arrays[0]).squeeze(2).transpose(1, 2)
    valid = torch.from_numpy(arrays[1]).reshape(1, 128)
    additive = (1 - valid[:, None, None, :]) * -32
    hidden = auto.language_model(inputs_embeds=x, attention_mask={
        'full_attention': additive, 'sliding_attention': additive}).last_hidden_state
    mean = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
    return torch.nn.functional.normalize(mean, p=2, dim=1).numpy().reshape(768)

runtime = None
try:
    save()
    gate_path = root / 'bpu/results/vision_split2_full_board_gate/summary.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'complete' and gate['joint_board_gate_passed']
    assert (root / 'bpu/logs/multimodal_repair/vision_split2_full_board.exit').read_text().strip() == '0'
    assert gate['manifest'] == manifest['gate_manifest'], 'Candidate provenance mismatch'
    for spec in manifest['gate_manifest']['vision'] + [manifest['gate_manifest']['fusion']]:
        assert sha(root / spec['path']) == spec['sha256']
    assert sha(root / manifest['caption_reference']) == manifest['caption_reference_sha256']
    data = np.load(root / manifest['gate_manifest']['data'])
    assert sha(root / manifest['gate_manifest']['data']) == manifest['gate_manifest']['data_sha256']
    caption_refs = np.load(root / manifest['caption_reference'])
    for case in manifest['images']:
        assert sha(root / case['source']) == case['source_sha256']
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    model = SentenceTransformer(str(root / 'model'), device='cpu', local_files_only=True,
        config_kwargs={'audio_config': None, **({'vision_config': None} if args.mode == 'bpu' else {})},
        model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'}).eval()
    auto = model[0].auto_model
    base = prepare(model, {'image': Image.open(root / manifest['images'][0]['source']).convert('RGB')})
    keep = ~(base['positions'] == -1).all(-1)[0].numpy()
    vision_cpu = None
    if args.mode == 'cpu':
        with torch.no_grad():
            vision_cpu = FixedVision(auto, base['positions'][:, keep]).eval()
    report['initial_model_load_seconds'] = time.perf_counter() - start
    save()

    def encode(kind, index):
        global runtime
        begin = time.perf_counter()
        stages = []
        if kind == 'text':
            item = index % len(manifest['gate_manifest']['captions'])
            p = prepare(model, manifest['gate_manifest']['captions'][item])
            arrays = fuse(p)
            reference = caption_refs[item]
        else:
            item = index % len(manifest['images'])
            case = manifest['images'][item]
            image = Image.open(root / case['source']).convert('RGB')
            if case['letterbox']:
                image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
            value = {'image': image}
            if kind == 'mixed':
                value['text'] = case['mixed_text'] + ' <|image|>'
            p = prepare(model, value)
            assert torch.equal(p['positions'], base['positions']), 'Unsupported position grid'
            pixels = np.ascontiguousarray(p['pixels'][:, :, :, keep])
            assert np.array_equal(pixels, data[f'pixels_{item}']), 'Input preprocessing drift'
            if args.mode == 'cpu':
                with torch.no_grad():
                    raw = vision_cpu(torch.from_numpy(pixels)).numpy()
            else:
                raw = pixels
                for chunk, spec in enumerate(manifest['gate_manifest']['vision']):
                    start = time.perf_counter()
                    runtime = Runtime(root / spec['path'], [(1, 768, 1, 576)])
                    loaded = time.perf_counter()
                    raw, ms = runtime.run([raw])
                    runtime.close()
                    runtime = None
                    stages.append({'chunk': chunk, 'core_ms': ms,
                        'load_ms': (loaded-start)*1000, 'total_ms': (time.perf_counter()-start)*1000})
            arrays = fuse(p, raw.squeeze(2).transpose(0, 2, 1)[0])
            reference = data[f'ref_{item}_{kind}']
        if args.mode == 'cpu':
            with torch.no_grad():
                pred = cpu_fusion(auto, arrays)
        else:
            start = time.perf_counter()
            spec = manifest['gate_manifest']['fusion']
            runtime = Runtime(root / spec['path'], [(1, 512, 1, 128), (1, 1, 1, 128)])
            loaded = time.perf_counter()
            raw, ms = runtime.run(arrays)
            runtime.close()
            runtime = None
            pred = raw.reshape(768)
            pred /= np.linalg.norm(pred)
            stages.append({'chunk': 'fusion', 'core_ms': ms,
                'load_ms': (loaded-start)*1000, 'total_ms': (time.perf_counter()-start)*1000})
        assert np.isfinite(pred).all()
        wall = (time.perf_counter()-begin)*1000
        cosine = float(pred @ reference)
        assert cosine >= (.9999 if args.mode == 'cpu' else .95), (kind, item, cosine)
        return {'kind': kind, 'item': item, 'tokens': p['tokens'],
            'wall_ms': wall, 'cosine': cosine, 'stages': stages}, pred

    for kind in ['text', 'image', 'mixed']:
        cold, pred = encode(kind, 0)
        cold['cold'] = True
        report['samples'].append(cold)
        np.save(out / f'{kind}_cold_embedding.npy', pred)
        save()
        for i in range(args.repeats):
            row, pred = encode(kind, i)
            row.update(cold=False, repeat=i)
            report['samples'].append(row)
            if i % 10 == 0:
                row['state'] = state()
            np.save(out / f'{kind}_{i:03d}_embedding.npy', pred)
            save()
        report[kind] = stats([c['wall_ms'] for c in report['samples'] if c['kind'] == kind and not c['cold']])
        save()
    report.update(status='complete', finished_at=time.time(), state_after=state(),
        all_sample_accuracy_passed=True, end_to_end_100_tested=args.mode=='bpu' and args.repeats>=100)
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
finally:
    if runtime is not None:
        runtime.close()
