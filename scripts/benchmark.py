"""Offline device benchmarks. All timed encode calls include preprocessing."""
import argparse
import json
import os
import platform
import resource
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import numpy as np
import psutil
import torch
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

def state():
    temperatures = {}
    for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            temperatures[p.parent.name] = int(p.read_text()) / 1000
        except (OSError, ValueError):
            pass
    frequencies = []
    for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'):
        try:
            frequencies.append(int(p.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return {'temperature_c': temperatures, 'cpu_frequency_mhz': frequencies,
            'rss_mb': psutil.Process().memory_info().rss / 1e6,
            'available_memory_mb': psutil.virtual_memory().available / 1e6,
            'load_average': os.getloadavg()}

def validate(emb, count):
    assert emb.shape == (count, 768), emb.shape
    assert np.isfinite(emb).all(), 'Non-finite embedding'
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4), 'Non-unit embedding'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--profile', choices=['smoke', 'screen', 'full', 'image'], default='smoke')
    p.add_argument('--repeats', type=int, default=7)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = ROOT / 'results' / f'{args.profile}_t{args.threads}.json'
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'profile': args.profile,
              'threads': args.threads, 'device': 'cpu', 'precision': 'float32',
              'model': 'google/embeddinggemma-2', 'revision': '914f7f89142e33e77833254d9c9b90c3cef7303b',
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'transformers': transformers.__version__, 'sentence_transformers': sentence_transformers.__version__},
              'state_before': state(), 'cases': []}
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.wait(0.25):
            samples.append(state())
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    t = time.perf_counter()
    cfg = {'audio_config': None}
    if args.profile != 'image':
        cfg['vision_config'] = None
    model = SentenceTransformer(str(ROOT / 'model'), device='cpu', local_files_only=True,
                               config_kwargs=cfg, model_kwargs={'dtype': torch.float32})
    model.eval()
    model.max_seq_length = 2048  # Bound this experiment; model supports 8192.
    result.update(load_seconds=time.perf_counter() - t, parameter_count=sum(x.numel() for x in model.parameters()),
                  dtype=str(model.dtype), state_after_load=state(), attention_implementation=model[0].auto_model.config.text_config._attn_implementation)
    print('LOADED', result['load_seconds'], result['parameter_count'], result['dtype'], flush=True)
    save()

    def measured(name, inputs, prompt=None):
        def encode():
            return model.encode(inputs, batch_size=len(inputs), prompt_name=prompt,
                                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        t = time.perf_counter()
        emb = encode()
        first = time.perf_counter() - t
        validate(emb, len(inputs))
        encode()  # second warmup, neither warmup included in steady-state statistics
        raw = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            emb = encode()
            raw.append(time.perf_counter() - t)
            validate(emb, len(inputs))
        case = {'name': name, 'batch_size': len(inputs), 'first_call_s': first, 'warmups': 2,
                'repeats': args.repeats, 'raw_seconds': raw, 'median_ms': float(np.median(raw) * 1000),
                'p95_ms': float(np.percentile(raw, 95) * 1000),
                'texts_per_second': len(inputs) / float(np.median(raw)), 'state_after': state()}
        if prompt:
            prefix = model.prompts[prompt]
            case['actual_tokens'] = [len(model.tokenizer.encode(prefix + x)) for x in inputs]
        result['cases'].append(case)
        save()
        print('CASE', name, json.dumps(case, ensure_ascii=False), flush=True)
        return emb

    if args.profile == 'image':
        from PIL import Image, ImageDraw
        images = []
        for color in ['red', 'blue', 'green']:
            im = Image.new('RGB', (224, 224), 'white')
            ImageDraw.Draw(im).rectangle((30, 30, 194, 194), fill=color)
            im.save(ROOT / 'results' / f'{color}_square.png')
            images.append({'image': im})
        image_embeddings = measured('synthetic_color_images', images)
        query_embeddings = measured('image_text_queries', ['a red square', 'a blue square', 'a green square'], 'SearchQuery')
        scores = query_embeddings @ image_embeddings.T
        result['image_smoke'] = {'fixture': 'synthetic colored squares, not a real image benchmark',
                                 'scores': scores.tolist(), 'top1': scores.argmax(axis=1).tolist()}
    else:
        measured('short_zh', ['如何排查机器人摄像头无法识别的问题？'], 'SearchQuery')
        measured('short_en', ['How do I diagnose a robot camera connection failure?'], 'SearchQuery')
        if args.profile in ['screen', 'full']:
            lengths = [32, 128] if args.profile == 'screen' else [32, 128, 256]
            seed = 'The robot inspects industrial products using a camera and detects surface defects. '
            ids = model.tokenizer.encode(seed * 300, add_special_tokens=False)
            for length in lengths:
                text = model.tokenizer.decode(ids[:length], skip_special_tokens=True)
                measured(f'en_content_tokens_{length}_b1', [text], 'Document')
                if args.profile == 'full' and length in [32, 128]:
                    measured(f'en_content_tokens_{length}_b4', [text] * 4, 'Document')
        if args.profile in ['smoke', 'full']:
            docs = [
                '排查摄像头连接时应检查USB线缆、设备节点和供电，确认摄像头是否被系统识别。',
                '电机无法旋转时，需要检查驱动器使能、电源和控制信号。',
                '温度传感器通过I2C总线读取环境温度，单位为摄氏度。',
                '部署神经网络模型需要校验权重文件，安装推理依赖并测量内存占用。',
                'Mars is known as the Red Planet because iron minerals give its surface a reddish color.',
                'The northern lights occur when charged particles from the sun interact with the atmosphere.',
                'A database index speeds up queries by helping locate rows without scanning every record.',
                'Bread is made by mixing flour, water and yeast and baking the dough.']
            queries = ['摄像头插上以后系统看不到设备，应该检查什么？', '电机不动怎么排查？',
                       '怎样通过总线测量周围的温度？', '模型部署之前如何检查文件和运行环境？',
                       'Which planet is called the Red Planet?', 'What causes the northern lights?',
                       '如何加快数据库查询？', 'How is bread prepared?']
            de = measured('retrieval_documents_b8', docs, 'Document')
            qe = measured('retrieval_queries_b8', queries, 'SearchQuery')
            quality = {}
            for dim in [768, 512, 256, 128]:
                d = de[:, :dim] / np.linalg.norm(de[:, :dim], axis=1, keepdims=True)
                q = qe[:, :dim] / np.linalg.norm(qe[:, :dim], axis=1, keepdims=True)
                scores = q @ d.T
                ranks = np.argsort(-scores, axis=1)
                quality[str(dim)] = {'top1_accuracy': float(np.mean(ranks[:, 0] == np.arange(8))),
                                     'mrr': float(np.mean([1 / (np.where(ranks[i] == i)[0][0] + 1) for i in range(8)])),
                                     'ranking': ranks.tolist(), 'scores': scores.tolist()}
            result['retrieval_smoke'] = {'documents': docs, 'queries': queries, 'expected_indices': list(range(8)),
                                        'metrics': quality, 'note': '8 authored queries; smoke test, not MTEB or domain accuracy'}
    stop.set()
    monitor_thread.join(timeout=2)
    result['state_end'] = state()
    result['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    result['resource_samples'] = samples
    result['status'] = 'complete'
    save()
    print('COMPLETE', output, flush=True)

if __name__ == '__main__':
    main()
