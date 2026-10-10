"""Six-image scheduling probe; amortized throughput is not single-image latency."""
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
parser.add_argument('--warm-images', type=int, required=True)
args = parser.parse_args()
assert args.warm_images > 0
assert args.repeats > 0
root = Path(__file__).resolve().parents[1]
manifest = json.loads(Path(args.manifest).read_text())
out = root / ('bpu/results/mlp_text_core_probe_' + args.mode)
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
fusion_runtime = None
report['strategy'] = 'resident_fusion_batch6_by_vision_chunk'
report['batch_size'] = 6
report['warm_target_images_per_kind'] = args.warm_images
report['strategy'] = 'batch6_' + args.mode + ('_resident_fusion' if args.mode == 'bpu' else '_fp32_resident_sequential')
report['batches'] = []
report['note'] = 'Each vision chunk loaded once per six-image batch; fusion held resident. All source reads, preprocessing, copies, normalization and chunk load/free counted in batch wall. Amortized throughput is not per-request latency.'
try:
    save()
    gate_path = root / 'bpu/results/mlp_boundary_full_board_gate/summary.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'complete' and gate['joint_board_gate_passed']
    assert (root / 'bpu/logs/multimodal_repair/mlp_full_board_gate.exit').read_text().strip() == '0'
    assert len(gate['cores'])==8 and all(c['core']['count']==100 for c in gate['cores'])
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
        global runtime, fusion_runtime
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
            core_start = time.perf_counter()
            with torch.no_grad():
                pred = cpu_fusion(auto, arrays)
            stages.append({'chunk': 'fusion_cpu', 'core_ms': (time.perf_counter()-core_start)*1000})
        else:
            start = time.perf_counter()
            spec = manifest['gate_manifest']['fusion']
            if fusion_runtime is None:
                fusion_runtime = Runtime(root / spec['path'], [(1, 512, 1, 128), (1, 1, 1, 128)])
                loaded = time.perf_counter()
                load_ms = (loaded-start)*1000
                report['initial_fusion_load_seconds'] = loaded-start
            else:
                load_ms = 0.0
            raw, ms = fusion_runtime.run(arrays)
            pred = raw.reshape(768)
            pred /= np.linalg.norm(pred)
            stages.append({'chunk': 'fusion', 'core_ms': ms,
                'load_ms': load_ms, 'total_ms': (time.perf_counter()-start)*1000})
        assert np.isfinite(pred).all()
        wall = (time.perf_counter()-begin)*1000
        cosine = float(pred @ reference)
        assert cosine >= (.9999 if args.mode == 'cpu' else .95), (kind, item, cosine)
        return {'kind': kind, 'item': item, 'tokens': p['tokens'],
            'wall_ms': wall, 'cosine': cosine, 'stages': stages}, pred

    def encode_batch(kind, size):
        global runtime
        begin = time.perf_counter()
        prepared, raws = [], []
        for item, case in enumerate(manifest['images'][:size]):
            image = Image.open(root / case['source']).convert('RGB')
            if case['letterbox']:
                image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
            value = {'image': image}
            if kind == 'mixed':
                value['text'] = case['mixed_text'] + ' <|image|>'
            p = prepare(model, value)
            assert torch.equal(p['positions'], base['positions'])
            pixels = np.ascontiguousarray(p['pixels'][:, :, :, keep])
            assert np.array_equal(pixels, data[f'pixels_{item}'])
            prepared.append(p)
            raws.append(pixels)
        assert len(raws) == size
        stages = []
        if args.mode == 'cpu':
            cpu_core = []
            for item in range(len(raws)):
                start = time.perf_counter()
                with torch.no_grad():
                    raws[item] = vision_cpu(torch.from_numpy(raws[item])).numpy()
                cpu_core.append((time.perf_counter()-start)*1000)
            stages.append({'chunk': 'vision_cpu', 'load_ms': 0.0, 'core_ms_each': cpu_core})
        else:
            for chunk, spec in enumerate(manifest['gate_manifest']['vision']):
                start = time.perf_counter()
                runtime = Runtime(root / spec['path'], [(1, 768, 1, 576)])
                loaded = time.perf_counter()
                core = []
                for item in range(len(raws)):
                    raws[item], ms = runtime.run([raws[item]])
                    core.append(ms)
                runtime.close()
                runtime = None
                stages.append({'chunk': chunk, 'load_ms': (loaded-start)*1000,
                               'core_ms_each': core, 'total_ms': (time.perf_counter()-start)*1000})
        rows, preds, fusion_core = [], [], []
        for item, (p, raw) in enumerate(zip(prepared, raws)):
            arrays = fuse(p, raw.squeeze(2).transpose(0, 2, 1)[0])
            if args.mode == 'cpu':
                start = time.perf_counter()
                with torch.no_grad():
                    pred = cpu_fusion(auto, arrays)
                ms = (time.perf_counter()-start)*1000
            else:
                output, ms = fusion_runtime.run(arrays)
                pred = output.reshape(768)
                pred /= np.linalg.norm(pred)
            cosine = float(pred @ data[f'ref_{item}_{kind}'])
            assert np.isfinite(pred).all() and cosine >= (.9999 if args.mode == 'cpu' else .95), (kind, item, cosine)
            ranking = np.argsort(-(board_captions @ pred)).tolist()
            expected = manifest['gate_manifest']['images'][item]['expected_caption_indices']
            assert ranking[0] in expected, (kind, item, ranking, expected)
            rows.append({'kind': kind, 'item': item, 'tokens': p['tokens'],
                         'cosine': cosine, 'ranking_board_captions': ranking, 'top1_correct': True})
            preds.append(pred)
            fusion_core.append(ms)
        stages.append({'chunk': 'fusion_' + args.mode, 'load_ms': 0.0, 'core_ms_each': fusion_core})
        wall = (time.perf_counter()-begin)*1000
        return {'kind': kind, 'batch_wall_ms': wall, 'amortized_ms_per_image': wall/size, 'batch_size': size,
                'stages': stages, 'items': rows, 'state': state()}, preds

    # Actual board caption vectors for retrieval, plus cold startup latency.
    board_captions = []
    cold, pred = encode('text', 0)
    cold['cold'] = True
    report['samples'].append(cold)
    np.save(out / 'text_cold_embedding.npy', pred)
    for i in range(max(8, args.repeats)):
        row, pred = encode('text', i)
        row.update(cold=False, repeat=i)
        report['samples'].append(row)
        if i < 8:
            board_captions.append(pred.copy())
        np.save(out / f'text_{i:03d}_embedding.npy', pred)
        save()
    board_captions = np.stack(board_captions)
    report['text'] = stats([c['wall_ms'] for c in report['samples'] if not c['cold']])
    save()
    for kind in []:
        for repeat in range(-1, (args.warm_images+5)//6):
            size = 6 if repeat == -1 else min(6, args.warm_images-repeat*6)
            batch, preds = encode_batch(kind, size)
            batch.update(cold=repeat == -1, repeat=repeat)
            report['batches'].append(batch)
            for item, pred in enumerate(preds):
                np.save(out / f'{kind}_batch_{repeat+1:03d}_item_{item}_embedding.npy', pred)
            save()
        warm = [b for b in report['batches'] if b['kind'] == kind and not b['cold']]
        report[kind] = {'warm_batches': len(warm), 'warm_images': sum(b['batch_size'] for b in warm),
                        'batch_wall': stats([b['batch_wall_ms'] for b in warm]),
                        'amortized_per_image': stats([b['amortized_ms_per_image'] for b in warm]),
                        'weighted_ms_per_image': sum(b['batch_wall_ms'] for b in warm) / sum(b['batch_size'] for b in warm)}
        save()
    report.update(status='complete', finished_at=time.time(), state_after=state(),
                  all_sample_accuracy_passed=True, all_image_retrieval_passed=True,
                  end_to_end_100_tested=args.mode == 'bpu' and args.repeats >= 100 and args.warm_images >= 100,
                  full_acceptance_passed=False, diagnostic_scope='CPU text core only, eight warm captions')
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
finally:
    if fusion_runtime is not None:
        fusion_runtime.close()
    if runtime is not None:
        runtime.close()
