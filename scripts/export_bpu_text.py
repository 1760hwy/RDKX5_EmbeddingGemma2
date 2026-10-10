"""Export the original text network at fixed length 32; token lookup stays on CPU.

The wrapper supplies an equivalent padding mask directly, avoiding dynamic
mask-building code during ONNX tracing. At length 32 the 512-token sliding
window covers the complete sequence. Validate against the original encoder.
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import onnx
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bpu'
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
model = SentenceTransformer(str(ROOT / 'model'), device='cpu', local_files_only=True,
    config_kwargs={'vision_config': None, 'audio_config': None},
    model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'})
model.max_seq_length = 32
model.eval()
language = model[0].auto_model.language_model

class TextCore(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs_embeds, attention_mask):
        # NCHW feature map boundaries; internal transformer uses [N,L,C].
        x = inputs_embeds.squeeze(2).transpose(1, 2)
        valid = attention_mask.reshape(1, 32)
        additive = (1.0 - valid[:, None, None, :]) * -1e9
        h = self.network(inputs_embeds=x,
            attention_mask={'full_attention': additive, 'sliding_attention': additive}).last_hidden_state
        mean = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return torch.nn.functional.normalize(mean, p=2, dim=1).reshape(1, 768, 1, 1)

wrapper = TextCore(language).eval()
source = json.loads((ROOT / 'results/smoke_t4.json').read_text())['retrieval_smoke']
cases = [(s, 'Document') for s in source['documents']] + [(s, 'SearchQuery') for s in source['queries']]
references, prepared, comparisons = [], [], []
for i, (text, prompt) in enumerate(cases):
    features = model.tokenize([model.prompts[prompt] + text])
    ids, mask = features['input_ids'], features['attention_mask']
    padding = 32 - ids.shape[1]
    ids = torch.nn.functional.pad(ids, (0, padding), value=0)
    mask = torch.nn.functional.pad(mask, (0, padding), value=0)
    with torch.no_grad():
        embedded = model[0].auto_model.get_input_embeddings()(ids).transpose(1, 2).unsqueeze(2)
        valid = mask.float().reshape(1, 1, 1, 32)
        result = wrapper(embedded, valid).reshape(1, 768).numpy()
        reference = model.encode([text], prompt_name=prompt, normalize_embeddings=True, show_progress_bar=False)
    cosine = float(np.sum(result * reference))
    error = float(np.max(np.abs(result-reference)))
    assert cosine > 0.99999 and error < 1e-4, (i, cosine, error)
    for name, array in [('inputs_embeds', embedded.numpy()), ('attention_mask', valid.numpy())]:
        folder = OUT / 'calibration' / name
        folder.mkdir(parents=True, exist_ok=True)
        array.astype(np.float32).tofile(folder / f'{i:03d}.bin')
    prepared.append((embedded, valid))
    references.append(reference[0])
    comparisons.append({'case': i, 'tokens': int(mask.sum()), 'cosine': cosine, 'max_abs_error': error})
    print('WRAPPER_VALIDATED', comparisons[-1], flush=True)

path = OUT / 'models/text_core_l32.onnx'
start = time.perf_counter()
torch.onnx.export(wrapper, prepared[0], str(path), input_names=['inputs_embeds', 'attention_mask'],
    output_names=['embedding'], opset_version=11, dynamo=False, do_constant_folding=True)
graph = onnx.load(str(path))
onnx.checker.check_model(graph)
options = ort.SessionOptions()
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
onnx_comparisons = []
for i, (x, mask) in enumerate(prepared):
    result = session.run(None, {'inputs_embeds': x.numpy(), 'attention_mask': mask.numpy()})[0].reshape(768)
    cosine = float(result @ references[i])
    error = float(np.max(np.abs(result-references[i])))
    assert cosine > 0.99999 and error < 1e-4, (i, cosine, error)
    onnx_comparisons.append({'case': i, 'cosine': cosine, 'max_abs_error': error})
np.save(OUT / 'results/reference_embeddings.npy', np.asarray(references))
report = {'status': 'complete', 'sequence_length': 32, 'lookup_on_cpu': True,
    'export_seconds': time.perf_counter()-start, 'onnx_bytes': path.stat().st_size,
    'operators': dict(Counter(node.op_type for node in graph.graph.node)),
    'wrapper_comparisons': comparisons, 'onnx_comparisons': onnx_comparisons,
    'cases': [{'text': s, 'prompt': p} for s,p in cases],
    'calibration_note': '16 authored examples only; conversion feasibility and small retrieval check, not production calibration'}
(OUT / 'results/export_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('ONNX_EXPORT_VALIDATED', report['operators'], flush=True)
