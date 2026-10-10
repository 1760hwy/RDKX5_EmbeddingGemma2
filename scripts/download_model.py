"""Download a pinned official snapshot; verify sizes and LFS SHA256."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPO = 'google/embeddinggemma-2'
REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model'
OUT.mkdir(exist_ok=True)
metadata = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60))
(ROOT / 'logs').mkdir(exist_ok=True)
(ROOT / 'logs/model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
manifest = []
for item in metadata['siblings']:
    name = item['rfilename']
    if name == '.gitattributes':
        continue
    dest = OUT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('size')
    if not dest.exists() or dest.stat().st_size != expected:
        partial = dest.with_suffix(dest.suffix + '.partial')
        for attempt in range(5):
            try:
                # A fresh URL avoids stale signed CDN redirects on retries.
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true&t={int(time.time())}'
                with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as f:
                    count, last = 0, time.monotonic()
                    while chunk := response.read(8 * 1024 * 1024):
                        f.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last > 15:
                            print(f'{name}: {count / 1e6:.1f} MB', flush=True)
                            last = time.monotonic()
                if expected is not None and partial.stat().st_size != expected:
                    raise ValueError(f'size mismatch: {name}')
                partial.replace(dest)
                break
            except Exception as exc:
                print(f'retry {attempt + 1}: {name}: {exc}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(2)
    digest = hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        h = hashlib.sha256()
        with dest.open('rb') as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        digest = h.hexdigest()
    if item.get('lfs', {}).get('sha256') and digest != item['lfs']['sha256']:
        raise ValueError(f'SHA256 mismatch: {name}')
    manifest.append({'path': name, 'size': dest.stat().st_size, 'sha256': digest})
    print(f'verified {name}: {dest.stat().st_size} bytes', flush=True)
(ROOT / 'logs/model_manifest.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'files': manifest}, indent=2), encoding='utf-8')
