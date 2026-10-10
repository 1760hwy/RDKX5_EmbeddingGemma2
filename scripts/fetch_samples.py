"""Fetch the examples referenced by Google's multimodal guide; keep provenance."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'samples'
OUT.mkdir(exist_ok=True)
sources = {
    'GoldenGate.png': 'https://raw.githubusercontent.com/google-gemma/cookbook/main/apps/sample-data/GoldenGate.png',
    'kitchen_painting.jpg': 'https://raw.githubusercontent.com/google-gemma/cookbook/main/apps/sample-data/kitchen_painting.jpg',
    'roses-are.wav': 'https://ai.google.dev/gemma/docs/audio/roses-are.wav',
    'cat_dozing.mp4': 'https://raw.githubusercontent.com/bebechien/gemma/main/videos/cat_dozing.mp4',
}
manifest = []
for name, url in sources.items():
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    (OUT / name).write_bytes(content)
    manifest.append({'file': name, 'url': url, 'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    print(name, len(content), flush=True)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
