"""Evaluate saved quantized vision features through the original fusion model.

Software diagnosis only. Saves real fused inputs for a longer BPU export;
neither intermediate feature accuracy nor this test constitutes board acceptance.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bpu/results/multimodal_semantic_diagnosis'
OUT.mkdir(exist_ok=True)
REPORT = OUT / 'summary.json'
report = {'status': 'running', 'started_at': time.time(), 'device': 'host_cpu',
          'note': 'Original FP32 language fusion with saved quantized vision features; not board evidence',
          'image_max_soft_tokens': 70, 'cases': []}

def save():
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    save()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(ROOT/'model'), device='cpu', local_files_only=True,
        config_kwargs={'audio_config': None},
        model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'})
    model.eval()
    auto = model[0].auto_model
    original_method = auto.get_image_features
    refs = np.load(ROOT/'bpu/results/vision_reference_features.npy')
    quant = np.load(ROOT/'bpu/models/vision_a16_w8_sm8/calibration_predictions.npy')
    assert refs.shape == quant.shape == (2, 64, 512)
    captions = ['The Golden Gate Bridge in San Francisco.', 'A painting of a kitchen interior.',
                'A cat dozing on a sofa.', 'A sports car racing on a highway.',
                '旧金山的金门大桥。', '一幅描绘厨房内部的画。', '一只正在打瞌睡的猫。', '高速公路上飞驰的跑车。']
    text = model.encode(captions, normalize_embeddings=True, show_progress_bar=False)
    np.save(OUT/'caption_embeddings.npy', text)
    report['captions'] = captions
    processing = {'image': {'max_soft_tokens': 70}}
    for i, name in enumerate(['GoldenGate.png', 'kitchen_painting.jpg']):
        image = Image.open(ROOT/'samples'/name).convert('RGB')
        prepared = model[0].preprocess([{'image': image}], processing_kwargs=processing)
        pixels = prepared['pixel_values'].transpose(1, 2).unsqueeze(2).numpy()
        calibration = np.fromfile(ROOT/f'bpu/calibration/vision_pixels/{i:03d}.bin', np.float32).reshape(pixels.shape)
        assert np.array_equal(pixels, calibration), 'Saved features belong to different processed pixels'
        for mixed in [False, True]:
            input_case = {'image': image}
            if mixed:
                input_case['text'] = captions[i] + ' <|image|>'
            captured = {}
            def capture(module, args, kwargs):
                captured['inputs_embeds'] = kwargs['inputs_embeds'].detach().numpy().copy()
                captured['attention_mask'] = kwargs['attention_mask'].detach().numpy().copy()
            handle = auto.language_model.register_forward_pre_hook(capture, with_kwargs=True)
            def encode():
                return model.encode([input_case], normalize_embeddings=True,
                    show_progress_bar=False, processing_kwargs=processing)[0]
            # Verify saved FP32 feature substitution against the untouched model.
            auto.get_image_features = original_method
            original = encode()
            def inject(array):
                def replacement(pixel_values, image_position_ids=None, **kwargs):
                    assert pixel_values.shape[0] == 1
                    return SimpleNamespace(pooler_output=(torch.from_numpy(array[i]),))
                auto.get_image_features = replacement
            inject(refs)
            baseline = encode()
            assert float(original @ baseline) > 0.99999
            label = f'{i:03d}_' + ('mixed' if mixed else 'image')
            np.savez(OUT/f'{label}_fusion_inputs.npz', **captured)
            inject(quant)
            candidate = encode()
            np.savez(OUT/f'{label}_quantized_vision_fusion_inputs.npz', **captured)
            handle.remove()
            scores, reference_scores = candidate @ text.T, baseline @ text.T
            expected = [i, i+4]
            row = {'image': name, 'mixed_text_image': mixed,
                   'tokens': int(captured['attention_mask'].sum()),
                   'embedding_cosine': float(candidate @ baseline),
                   'reference_substitution_cosine': float(original @ baseline),
                   'reference_scores': reference_scores.tolist(), 'candidate_scores': scores.tolist(),
                   'reference_rankings': np.argsort(-reference_scores).tolist(),
                   'candidate_rankings': np.argsort(-scores).tolist(),
                   'expected_caption_group': expected,
                   'candidate_top1_correct': int(np.argmax(scores)) in expected,
                   'reference_top1_correct': int(np.argmax(reference_scores)) in expected}
            report['cases'].append(row)
            np.savez(OUT/f'{label}_embeddings.npz', original=original, reference=baseline, candidate=candidate)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            save()
    auto.get_image_features = original_method
    report['minimum_embedding_cosine'] = min(row['embedding_cosine'] for row in report['cases'])
    report['initial_semantic_gate_passed'] = report['minimum_embedding_cosine'] >= .95 and all(
        row['candidate_top1_correct'] and row['reference_top1_correct'] for row in report['cases'])
    report.update(status='complete', finished_at=time.time())
    save()

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        report.update(status='failed', error=repr(error), finished_at=time.time())
        save()
        raise
