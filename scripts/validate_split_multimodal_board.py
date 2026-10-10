"""Real board eight-chunk vision/fusion gate and remaining core timings.

This gate does not claim a complete end-to-end benchmark or production value.
"""
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
from PIL import Image
from sentence_transformers import SentenceTransformer
from persistent_bpu import Runtime
from prepare_multimodal_inputs import prepare, fuse

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads(Path(args.manifest).read_text())
out = root / manifest['output']
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'summary.json').exists(), 'Refuse to overwrite experiment'
report = {'status': 'running', 'phase': 'hashes', 'started_at': time.time(),
          'full_acceptance_passed': False, 'end_to_end_100_tested': False,
          'manifest': manifest, 'vision_cases': [], 'text_cases': [],
          'multimodal_cases': [], 'cores': []}

def save():
    (out / 'summary.json').write_text(json.dumps(report, indent=2))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize(value):
    value = value.reshape(768)
    assert np.isfinite(value).all() and np.linalg.norm(value) > 0
    return value / np.linalg.norm(value)

def timing(values):
    a = np.asarray(values)
    return {'mean_ms': float(a.mean()), 'p95_ms': float(np.percentile(a, 95)),
            'min_ms': float(a.min()), 'max_ms': float(a.max()),
            'count': len(values), 'raw_ms': values}

runtime = None
try:
    save()
    for spec in manifest['vision'] + [manifest['fusion']]:
        assert digest(root / spec['path']) == spec['sha256'], spec['path']
    path = root / manifest['data']
    assert digest(path) == manifest['data_sha256']
    data = np.load(path)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(root / 'model'), device='cpu',
        local_files_only=True, config_kwargs={'vision_config': None, 'audio_config': None},
        model_kwargs={'dtype': torch.float32}).eval()
    report['phase'] = 'vision'
    save()
    prepared, latents, first_inputs = {}, {}, []
    for index, case in enumerate(manifest['images']):
        source = root / case['image']
        assert digest(source) == case['image_sha256']
        image = Image.open(source).convert('RGB')
        p = prepare(model, {'image': image})
        keep = ~(p['positions'] == -1).all(-1)[0].numpy()
        assert int(keep.sum()) == 576
        pixels = np.ascontiguousarray(p['pixels'][:, :, :, keep])
        assert np.array_equal(pixels, data[f'pixels_{index}']), 'Preprocessing mismatch'
        prepared[(index, 'image')] = p
        prepared[(index, 'mixed')] = prepare(model, {
            'image': image, 'text': case['mixed_text'] + ' <|image|>'})
        assert np.array_equal(prepared[(index, 'mixed')]['pixels'], p['pixels'])
        current, stages = pixels, []
        begin = time.perf_counter()
        for chunk, spec in enumerate(manifest['vision']):
            if index == 0:
                first_inputs.append(current.copy())
            start = time.perf_counter()
            runtime = Runtime(root / spec['path'], [(1, 768, 1, 576)])
            loaded = time.perf_counter()
            expected = (1, 512, 1, 64) if chunk == 7 else (1, 768, 1, 576)
            assert runtime.output_shape == expected
            current, ms = runtime.run([current])
            runtime.close()
            runtime = None
            stages.append({'chunk': chunk, 'core_ms': ms,
                           'load_ms': (loaded - start) * 1000,
                           'total_ms': (time.perf_counter() - start) * 1000})
        reference = data[f'latent_{index}']
        assert current.shape == reference.shape
        report['vision_cases'].append({'case': case['case'],
            'max_abs_error_vs_quantized': float(np.max(np.abs(current - reference))),
            'relative_rmse_vs_quantized': float(np.linalg.norm(current-reference) /
                                                np.linalg.norm(reference)),
            'sequential_vision_ms': (time.perf_counter() - begin) * 1000,
            'stages': stages})
        latents[index] = current.squeeze(2).transpose(0, 2, 1)[0]
        np.save(out / f"{case['case']}_vision.npy", current)
        save()
    report['phase'] = 'fusion'
    save()
    fusion = manifest['fusion']
    shapes = [(1, 512, 1, 128), (1, 1, 1, 128)]
    runtime = Runtime(root / fusion['path'], shapes)
    assert runtime.output_shape == (1, 768, 1, 1)
    captions = []
    for caption in manifest['captions']:
        raw, _ = runtime.run(fuse(prepare(model, caption)))
        captions.append(normalize(raw))
    captions = np.asarray(captions)
    np.save(out / 'board_caption_embeddings.npy', captions)
    text = []
    for index, case in enumerate(manifest['text']):
        raw, ms = runtime.run([data[f'text_x_{index}'], data[f'text_mask_{index}']])
        predicted = normalize(raw)
        text.append(predicted)
        report['text_cases'].append({'case': case['case'],
            'cosine': float(predicted @ data[f'text_ref_{index}']), 'core_ms': ms})
        save()
    text = np.asarray(text)
    np.save(out / 'board_text_embeddings.npy', text)
    for group in ['text_calibration', 'text_heldout']:
        indices = [i for i, c in enumerate(manifest['text']) if c['case'].startswith(group)]
        assert len(indices) == 16
        values = text[indices]
        ranking = np.argmax(values[8:] @ values[:8].T, axis=1)
        report[group] = {'top1': int(np.sum(ranking == np.arange(8))),
                         'total': 8, 'indices': ranking.tolist()}
    sample_fusion = None
    for index, case in enumerate(manifest['images']):
        for kind in ['image', 'mixed']:
            arrays = fuse(prepared[(index, kind)], latents[index])
            if sample_fusion is None:
                sample_fusion = arrays
            raw, ms = runtime.run(arrays)
            predicted = normalize(raw)
            scores = predicted @ captions.T
            ranking = np.argsort(-scores)
            label = case['case'] + '_' + kind
            report['multimodal_cases'].append({'case': label,
                'cosine': float(predicted @ data[f'ref_{index}_{kind}']),
                'caption_rankings': ranking.tolist(), 'caption_scores': scores.tolist(),
                'top1_correct': int(ranking[0]) in case['expected_caption_indices'],
                'fusion_core_ms': ms})
            np.save(out / f'{label}_embedding.npy', predicted)
            save()
    report['joint_board_gate_passed'] = (
        all(c['cosine'] >= .95 for c in report['text_cases']) and
        all(report[g]['top1'] == 8 for g in ['text_calibration', 'text_heldout']) and
        all(c['cosine'] >= .95 and c['top1_correct'] for c in report['multimodal_cases']))
    runtime.close()
    runtime = None
    save()
    assert report['joint_board_gate_passed'], 'Joint board semantic gate failed'
    report['phase'] = 'remaining_core_100'
    save()
    # Earlier four chunks already have 100-frame measurements at these same SHAs.
    for index, spec in enumerate(manifest['vision'][4:] + [fusion], start=4):
        arrays = [first_inputs[index]] if index < 8 else sample_fusion
        expected_shapes = [(1, 768, 1, 576)] if index < 8 else shapes
        begin = time.perf_counter()
        runtime = Runtime(root / spec['path'], expected_shapes)
        load_seconds = time.perf_counter() - begin
        expected, cold = runtime.run(arrays)
        core, wall = [], []
        for _ in range(100):
            begin = time.perf_counter()
            value, ms = runtime.run(arrays)
            wall.append((time.perf_counter() - begin) * 1000)
            core.append(ms)
            assert np.array_equal(value, expected), 'Non-deterministic output'
        runtime.close()
        runtime = None
        report['cores'].append({'chunk': index, 'load_seconds': load_seconds,
            'cold_core_ms': cold, 'core': timing(core), 'adapter_wall': timing(wall)})
        save()
    report.update(status='complete', finished_at=time.time(),
                  note='Board semantic gate/core timings only; full pipeline 100 frames and CPU baseline pending')
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
finally:
    if runtime is not None:
        runtime.close()
