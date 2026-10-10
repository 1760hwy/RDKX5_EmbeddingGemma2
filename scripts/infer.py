"""Reusable offline text embedding and retrieval CLI on RDK X5."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--query', required=True)
parser.add_argument('--documents', help='UTF-8 JSON file containing a non-empty list of strings')
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--dimension', type=int, choices=[128, 256, 512, 768], default=768)
parser.add_argument('--output', default=str(ROOT / 'results/inference_demo.json'))
args = parser.parse_args()
torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
t = time.perf_counter()
model = SentenceTransformer(str(ROOT / 'model'), device='cpu', local_files_only=True,
                           config_kwargs={'vision_config': None, 'audio_config': None},
                           model_kwargs={'dtype': torch.float32})
load_s = time.perf_counter() - t
t = time.perf_counter()
query = model.encode([args.query], prompt_name='SearchQuery', truncate_dim=args.dimension,
                     normalize_embeddings=True, show_progress_bar=False)[0]
assert np.isfinite(query).all()
result = {'model': 'google/embeddinggemma-2', 'device': 'cpu', 'dtype': str(model.dtype),
          'threads': args.threads, 'dimension': args.dimension, 'query': args.query,
          'load_seconds': load_s, 'query_seconds': time.perf_counter() - t,
          'embedding': query.tolist()}
if args.documents:
    docs = json.loads(Path(args.documents).read_text(encoding='utf-8'))
    if not isinstance(docs, list) or not docs or not all(isinstance(x, str) for x in docs):
        parser.error('--documents must contain a non-empty JSON list of strings')
    t = time.perf_counter()
    vectors = model.encode(docs, batch_size=4, prompt_name='Document', truncate_dim=args.dimension,
                           normalize_embeddings=True, show_progress_bar=False)
    scores = vectors @ query
    result['documents_seconds'] = time.perf_counter() - t
    result['ranking'] = [{'index': int(i), 'score': float(scores[i]), 'document': docs[i]}
                         for i in np.argsort(-scores)]
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'embedding'}, ensure_ascii=False, indent=2))
print(f'Saved {args.output}')
