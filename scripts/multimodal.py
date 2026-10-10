"""Real image/audio/video and mixed-input experiments, on the device CPU."""
import argparse
import json
import resource
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from benchmark import ROOT, state, validate, np, torch, SentenceTransformer
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['vision', 'audio', 'mixed'], required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    output = ROOT / 'results' / f'multimodal_{a.mode}_t{a.threads}.json'
    result = {'profile': f'multimodal_{a.mode}', 'mode': a.mode, 'threads': a.threads,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'device': 'cpu', 'dtype': 'float32',
              'cases': [], 'state_before': state(), 'model_revision': '914f7f89142e33e77833254d9c9b90c3cef7303b'}
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.wait(0.5):
            samples.append(state())
    threading.Thread(target=monitor, daemon=True).start()
    cfg = {'audio_config': None} if a.mode == 'vision' else {'vision_config': None} if a.mode == 'audio' else {}
    t = time.perf_counter()
    model = SentenceTransformer(str(ROOT / 'model'), device='cpu', local_files_only=True,
                               config_kwargs=cfg, model_kwargs={'dtype': torch.float32})
    model.eval()
    result.update(load_seconds=time.perf_counter() - t, parameter_count=sum(x.numel() for x in model.parameters()),
                  state_after_load=state(), torch_version=torch.__version__)
    print('LOADED', a.mode, result['load_seconds'], result['parameter_count'], flush=True)
    save()
    if state()['available_memory_mb'] < 700:
        raise RuntimeError('Less than 700 MB available after loading; stopping before inference')

    def measured(name, inputs, processing=None, detail=None):
        def encode():
            return model.encode(inputs, batch_size=len(inputs), normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True,
                                **({'processing_kwargs': processing} if processing else {}))
        t = time.perf_counter()
        emb = encode()
        first = time.perf_counter() - t
        validate(emb, len(inputs))
        raw = []
        for _ in range(a.repeats):
            t = time.perf_counter()
            emb = encode()
            raw.append(time.perf_counter() - t)
            validate(emb, len(inputs))
        case = {'name': name, 'batch_size': len(inputs), 'first_call_s': first, 'warmups': 1,
                'repeats': a.repeats, 'raw_seconds': raw, 'median_ms': float(np.median(raw) * 1000),
                'p95_ms': float(np.percentile(raw, 95) * 1000), 'texts_per_second': len(inputs) / float(np.median(raw)),
                'detail': detail, 'state_after': state()}
        result['cases'].append(case)
        save()
        print('CASE', json.dumps(case, ensure_ascii=False), flush=True)
        np.save(ROOT / 'results' / f'{a.mode}_{name}_embeddings.npy', emb)
        return emb

    image = Image.open(ROOT / 'samples/GoldenGate.png').convert('RGB')
    if a.mode == 'vision':
        kitchen = Image.open(ROOT / 'samples/kitchen_painting.jpg').convert('RGB')
        captions = ['The Golden Gate Bridge in San Francisco.', 'A painting of a kitchen interior.',
                    'A cat dozing on a sofa.', 'A sports car racing on a highway.',
                    '旧金山的金门大桥。', '一幅描绘厨房内部的画。', '一只正在打瞌睡的猫。', '高速公路上飞驰的跑车。']
        te = measured('image_video_text_candidates', captions)
        ie = measured('real_images_b2', [{'image': image}, {'image': kitchen}],
                      detail={'files': ['GoldenGate.png', 'kitchen_painting.jpg'], 'original_sizes': [image.size, kitchen.size], 'max_soft_tokens': 280})
        single = measured('goldengate_image_b1', [{'image': image}], detail={'max_soft_tokens': 280})
        result['image_retrieval'] = {'captions': captions, 'scores': (ie @ te.T).tolist(),
                                     'rankings': np.argsort(-(ie @ te.T), axis=1).tolist(),
                                     'expected_caption_groups': [[0, 4], [1, 5]],
                                     'batch_single_cosine': float(ie[0] @ single[0])}
        frames = [Image.open(f).convert('RGB') for f in sorted((ROOT / 'samples/prepared/video_frames').glob('*.png'))]
        assert len(frames) == 4, len(frames)
        ve = measured('real_video_4_frames', [{'video': frames}],
                      processing={'video': {'do_sample_frames': False, 'add_timestamps': False, 'max_soft_tokens': 140}},
                      detail={'source': 'cat_dozing.mp4', 'clip_seconds': 4, 'frames': 4, 'sampling_fps': 1,
                              'decode': 'ffmpeg before timing; predecoded frames', 'add_timestamps': False})
        result['video_retrieval'] = {'captions': captions, 'scores': (ve @ te.T).tolist(),
                                     'rankings': np.argsort(-(ve @ te.T), axis=1).tolist(), 'expected_caption_group': [2, 6]}
        mixed = measured('text_image_interleaved', [{'text': 'The Golden Gate Bridge in San Francisco. <|image|>', 'image': image}])
        result['text_image_retrieval'] = {'captions': captions, 'scores': (mixed @ te.T).tolist(),
                                          'rankings': np.argsort(-(mixed @ te.T), axis=1).tolist()}
    else:
        with wave.open(str(ROOT / 'samples/prepared/speech_16k.wav'), 'rb') as w:
            assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
            speech = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768
        seconds = len(speech) / 16000
        captions = ['a person speaking English', 'random static white noise', 'Roses are red, violets are blue.',
                    'A sports car racing on a highway.', '玫瑰是红色的，紫罗兰是蓝色的。', '一个人在说英语。', '一只猫在睡觉。']
        if a.mode == 'audio':
            te = measured('audio_text_candidates', captions)
            ae = measured('real_speech_audio', [{'audio': speech}], detail={'source': 'roses-are.wav', 'seconds': seconds, 'sample_rate': 16000, 'decode': 'wave read before timing'})
            result['audio_retrieval'] = {'captions': captions, 'scores': (ae @ te.T).tolist(),
                                         'rankings': np.argsort(-(ae @ te.T), axis=1).tolist(), 'expected_content_groups': [2, 4]}
        else:
            mixed = measured('text_image_audio_interleaved', [{
                'text': 'A bridge photograph <|image|> and spoken English verse <|audio|>.',
                'image': image, 'audio': speech}], detail={'audio_seconds': seconds, 'images': 1, 'purpose': 'combined-input validity, not a semantic accuracy test'})
            result['mixed_input'] = {'shape': list(mixed.shape), 'finite': bool(np.isfinite(mixed).all()), 'norms': np.linalg.norm(mixed, axis=1).tolist()}
    stop.set()
    result['state_end'] = state()
    result['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    result['resource_samples'] = samples
    result['status'] = 'complete'
    save()
    print('COMPLETE', output, flush=True)

if __name__ == '__main__':
    main()
