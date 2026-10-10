"""LAN API integration checks; invokes the real board and saves raw results."""
import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'bpu/results/web_ui_acceptance'
BASE = 'http://192.0.2.10:8088'
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OUT.mkdir(parents=True, exist_ok=True)

def request(path, body=None, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    try:
        with HTTP.open(req, timeout=15) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

def run(payload):
    code, job = request('/api/jobs', payload)
    assert code == 202, (code, job)
    for _ in range(180):
        _, state = request('/api/jobs/' + job['id'])
        if state['status'] in ('complete', 'failed'):
            return state
        time.sleep(1)
    raise TimeoutError(job)

for _ in range(90):
    _, status = request('/api/status')
    if status['status'] == 'ready':
        break
    assert status['status'] != 'failed', status
    time.sleep(1)
else:
    raise TimeoutError('model startup')

captions = ['The Golden Gate Bridge in San Francisco.', 'A painting of a kitchen interior.', 'A cat dozing on a sofa.']
refs = np.load(ROOT / 'bpu/vision_trim576/mlp_boundary_full16/full_board_inputs/data.npz')
caption_refs = np.load(ROOT / 'bpu/results/multimodal_semantic_diagnosis/caption_embeddings.npy')
cases = [
    ('text', {'text': captions[0]}, 'text_ref_0', captions[0]),
    ('image', {'sample': 'bridge'}, 'ref_0_image', captions[0]),
    ('mixed', {'sample': 'kitchen', 'text': captions[1]}, 'ref_1_mixed', captions[1]),
    ('upload', {'image': 'data:image/png;base64,' + base64.b64encode((ROOT / 'samples/prepared/video_frames/frame_01.png').read_bytes()).decode()}, 'ref_2_image', captions[2]),
]
rows = []
for name, payload, key, top in cases:
    state = run(dict(payload, candidates=captions))
    (OUT / (name + '.json')).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    assert state['status'] == 'complete', state
    r = state['result']
    vector = np.array(r['embedding'])
    reference = caption_refs[0] if name == 'text' else refs[key]
    cos = float(vector @ reference / np.linalg.norm(vector) / np.linalg.norm(reference))
    assert cos >= .95 and abs(np.linalg.norm(vector) - 1) < 1e-5
    assert r['ranking'][0]['text'] == top, r['ranking']
    rows.append({'case': name, 'fp32_cosine': cos, 'input_ms': r['input_timing']['wall_ms'], 'total_ms': r['total_ms'], 'top': top})
    print(rows[-1], flush=True)

invalid = [({'text': '', 'candidates': captions}, 400),
           ({'text': 'ok', 'candidates': captions, 'sample': ['bad']}, 400),
           ({'text': 'ok', 'candidates': captions, 'image': 'data:image/png;base64,???'}, 400)]
for payload, expected in invalid:
    assert request('/api/jobs', payload)[0] == expected
assert request('/api/jobs', {'text': 'ok', 'candidates': captions}, 'http://other.example')[0] == 403
bad = run({'candidates': captions, 'image': 'data:image/png;base64,' + base64.b64encode(b'not an image').decode()})
assert bad['status'] == 'failed' and '图片' in bad['error']
assert run({'text': captions[0], 'candidates': captions})['status'] == 'complete'
report = {'status': 'complete', 'real_board': True, 'cases': rows, 'invalid_requests_rejected': True, 'invalid_image_recovery': True, 'model_status': status}
(OUT / 'api_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('API_CHECKS_PASSED', flush=True)
