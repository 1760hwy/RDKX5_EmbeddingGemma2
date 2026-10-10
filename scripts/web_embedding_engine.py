"""Real, serial BPU embedding inference using the accepted candidate manifest."""
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch
from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer
from persistent_bpu import Runtime
from prepare_multimodal_inputs import prepare, fuse

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'bpu/vision_trim576/mlp_boundary_full16/full_board_inputs/pipeline_manifest.json'


class Engine:
    def __init__(self, progress):
        self.manifest = json.loads(MANIFEST.read_text())['gate_manifest']
        self.cache = OrderedDict()
        self.fusion = None
        progress('核验已验收模型文件')
        for spec in self.manifest['vision'] + [self.manifest['fusion']]:
            assert hashlib.sha256((ROOT / spec['path']).read_bytes()).hexdigest() == spec['sha256'], 'Model SHA mismatch'
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        progress('加载本地分词器与CPU查表模型')
        start = time.perf_counter()
        self.model = SentenceTransformer(str(ROOT / 'model'), device='cpu', local_files_only=True,
            config_kwargs={'audio_config': None, 'vision_config': None},
            model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'}).eval()
        base = prepare(self.model, {'image': Image.open(ROOT / 'samples/GoldenGate.png').convert('RGB')})
        self.keep = ~(base['positions'] == -1).all(-1)[0].numpy()
        self.positions = base['positions']
        assert int(self.keep.sum()) == 576
        progress('加载BPU融合模型，准备接受请求')
        self.fusion = Runtime(ROOT / self.manifest['fusion']['path'], [(1, 512, 1, 128), (1, 1, 1, 128)])
        self.load_seconds = time.perf_counter() - start

    def encode(self, text='', image=None, progress=lambda _: None):
        start = time.perf_counter()
        stages = []
        if image is None:
            value = text
        else:
            progress('图片方形留边与官方预处理')
            image = image.convert('RGB')
            if image.width != image.height:
                image = ImageOps.pad(image, (512, 512), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
            value = {'image': image}
            if text:
                value['text'] = text + ' <|image|>'
        try:
            p = prepare(self.model, value)
        except AssertionError as error:
            raise ValueError('输入不适合当前固定128 tokens配置，请缩短文本。' + str(error)) from error
        latent = None
        if image is not None:
            if not torch.equal(p['positions'], self.positions):
                raise ValueError('当前图片预处理网格不符合已验收模型。')
            latent = np.ascontiguousarray(p['pixels'][:, :, :, self.keep])
            for i, spec in enumerate(self.manifest['vision']):
                progress('视觉BPU推理 %d / 8' % (i + 1))
                begin = time.perf_counter()
                runtime = Runtime(ROOT / spec['path'], [(1, 768, 1, 576)])
                try:
                    latent, core_ms = runtime.run([latent])
                finally:
                    runtime.close()
                stages.append({'name': 'vision_%d' % i, 'core_ms': core_ms,
                               'load_infer_free_ms': (time.perf_counter() - begin) * 1000})
            latent = latent.squeeze(2).transpose(0, 2, 1)[0]
        progress('融合BPU推理与向量归一化')
        output, core_ms = self.fusion.run(fuse(p, latent))
        vector = output.reshape(768)
        vector /= np.linalg.norm(vector)
        assert np.isfinite(vector).all()
        stages.append({'name': 'fusion', 'core_ms': core_ms})
        return vector, {'wall_ms': (time.perf_counter() - start) * 1000, 'tokens': p['tokens'], 'stages': stages}

    def compare(self, text, image, candidates, progress):
        started = time.perf_counter()
        progress('计算输入向量')
        query, timing = self.encode(text, image, progress)
        vectors, hits = [], 0
        for i, candidate in enumerate(candidates):
            progress('候选描述 %d / %d' % (i + 1, len(candidates)))
            if candidate in self.cache:
                hits += 1
                vector = self.cache.pop(candidate)
            else:
                vector, _ = self.encode(candidate)
            self.cache[candidate] = vector
            while len(self.cache) > 128:
                self.cache.popitem(last=False)
            vectors.append(vector)
        scores = np.stack(vectors) @ query
        ranking = sorted([{'text': c, 'cosine': float(s)} for c, s in zip(candidates, scores)],
                         key=lambda row: row['cosine'], reverse=True)
        return {'kind': 'mixed' if image is not None and text else 'image' if image is not None else 'text',
                'ranking': ranking, 'embedding': query.tolist(), 'dimension': 768,
                'norm': float(np.linalg.norm(query)), 'input_timing': timing,
                'total_ms': (time.perf_counter() - started) * 1000,
                'candidate_cache_hits': hits, 'candidate_count': len(candidates),
                'backend': 'RDK X5 BPU · mlp_boundary_full16',
                'model_fusion_sha256': self.manifest['fusion']['sha256']}

    def close(self):
        if self.fusion is not None:
            self.fusion.close()
