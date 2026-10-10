"""FP32-only isolated lower-patch diagnostic; does not change deployment."""
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

root = Path(__file__).resolve().parents[1]
out = root / 'bpu/resolution_supported_diagnosis'
out.mkdir(exist_ok=True)
destination = out / 'summary.json'
assert not destination.exists(), 'Existing experiment: inspect before reuse'
manifest = json.loads((root/'bpu/vision_trim576/split2/full_board_inputs/manifest.json').read_text())
report = {'status': 'running', 'started_at': time.time(), 'board_tested': False,
    'quantized': False, 'weights_pruned': False, 'source_max_soft_tokens': 70,
    'note': 'Six images only; four correlated cat frames. Compare lower-resolution FP32 to original 70-token FP32. No board deployment modification.',
    'candidates': [], 'processor_do_resize': False, 'candidate_sides': [288, 336]}

def save():
    destination.write_text(json.dumps(report, indent=2))

try:
    save()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(root/'model'), device='cpu', local_files_only=True,
        config_kwargs={'audio_config': None},
        model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'}).eval()
    auto = model[0].auto_model
    captions = np.load(root/'bpu/results/multimodal_semantic_diagnosis/caption_embeddings.npy')
    data = np.load(root/manifest['data'])
    for side in [288, 336]:
        limit = 70
        folder = out / f'side_{side}'
        folder.mkdir(exist_ok=True)
        candidate = {'max_soft_tokens': limit, 'side': side, 'do_resize': False, 'cases': []}
        report['candidates'].append(candidate)
        options = {'image': {'max_soft_tokens': limit, 'do_resize': False}}
        for index, case in enumerate(manifest['images']):
            path = root/case['image']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case['image_sha256']
            image = Image.open(path).convert('RGB')
            # Task-owned reduction of the verified square input; original remains untouched.
            assert image.width == image.height, 'Square grid required by this diagnostic'
            image = image.resize((side, side), Image.Resampling.BICUBIC)
            for kind in ['image', 'mixed']:
                value = {'image': image}
                if kind == 'mixed':
                    value['text'] = case['mixed_text'] + ' <|image|>'
                features = model[0].preprocess([value], processing_kwargs=options)
                keep = ~(features['image_position_ids'] == -1).all(-1)[0]
                mask, _, _ = auto.get_placeholder_mask(features['input_ids'])
                captured = {}
                def hook(module, args, kwargs):
                    captured['inputs_embeds'] = kwargs['inputs_embeds'].detach().numpy().copy()
                    captured['attention_mask'] = kwargs['attention_mask'].detach().numpy().copy()
                handle = auto.language_model.register_forward_pre_hook(hook, with_kwargs=True)
                begin = time.perf_counter()
                try:
                    pred = model.encode([value], normalize_embeddings=True,
                        show_progress_bar=False, processing_kwargs=options)[0]
                finally:
                    handle.remove()
                ref = data[f'ref_{index}_{kind}']
                ranking = np.argsort(-(pred @ captions.T))
                label = case['case'] + '_' + kind
                np.savez(folder/f'{label}.npz', **captured,
                    pixels=features['pixel_values'][:, keep].transpose(1,2).unsqueeze(2).numpy(),
                    positions=features['image_position_ids'][:, keep].numpy(),
                    image_mask=mask.numpy(), reduced_fp32_embedding=pred, original_fp32_embedding=ref)
                row = {'case': label, 'valid_patches': int(keep.sum()),
                    'padded_patches': int(keep.numel()), 'image_slots': int(mask.sum()),
                    'fusion_tokens': int(features['input_ids'].shape[1]),
                    'cosine_to_original': float(pred @ ref), 'ranking': ranking.tolist(),
                    'top1_correct': int(ranking[0]) in case['expected_caption_indices'],
                    'host_fp32_encode_seconds': time.perf_counter()-begin}
                candidate['cases'].append(row)
                save()
                print(side, row, flush=True)
        candidate['minimum_cosine'] = min(c['cosine_to_original'] for c in candidate['cases'])
        candidate['all_retrieval_correct'] = all(c['top1_correct'] for c in candidate['cases'])
        # A stricter screen reserves error margin for subsequent quantization.
        candidate['eligible_for_quantization_diagnosis'] = (
            candidate['minimum_cosine'] >= .98 and candidate['all_retrieval_correct'])
        save()
    report.update(status='complete', finished_at=time.time())
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
