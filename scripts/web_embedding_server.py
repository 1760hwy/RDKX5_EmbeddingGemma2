"""LAN test UI, bounded job queue and a single real BPU worker; no new packages."""
import argparse
import base64
import binascii
import io
import json
import queue
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'web'
MAX_BODY = 8 * 1024 * 1024
SAMPLES = {'bridge': 'samples/GoldenGate.png', 'kitchen': 'samples/kitchen_painting.jpg',
           'cat': 'samples/prepared/video_frames/frame_01.png'}


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError('请求格式错误。')
    text = payload.get('text', '')
    candidates = payload.get('candidates')
    if not isinstance(text, str) or len(text) > 2000:
        raise ValueError('文本过长，请输入简短句子。')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        raise ValueError('候选描述须为1至12条，每行一条。')
    if any(not isinstance(c, str) or not c.strip() or len(c) > 500 for c in candidates):
        raise ValueError('候选描述不能为空或过长。')
    image = None
    sample = payload.get('sample')
    encoded = payload.get('image')
    if sample and encoded:
        raise ValueError('一次只能提供一张图片。')
    if sample:
        if not isinstance(sample, str) or sample not in SAMPLES:
            raise ValueError('未知示例图片。')
        image = (ROOT / SAMPLES[sample]).read_bytes()
    elif encoded:
        if not isinstance(encoded, str) or not encoded.startswith('data:image/'):
            raise ValueError('请选择图片文件。')
        try:
            image = base64.b64decode(encoded.split(',', 1)[1], validate=True)
        except (ValueError, IndexError, binascii.Error) as error:
            raise ValueError('图片数据无效。') from error
        if len(image) > 5 * 1024 * 1024:
            raise ValueError('图片最大5MB。')
    if not text.strip() and image is None:
        raise ValueError('请输入文本或选择图片。')
    return {'text': text.strip(), 'image_bytes': image, 'candidates': [c.strip() for c in candidates]}


class Service:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = OrderedDict()
        self.pending = queue.Queue(maxsize=3)
        self.state = {'status': 'loading', 'message': '正在启动模型', 'active_job': None}
        threading.Thread(target=self.worker, daemon=True).start()

    def update_state(self, message):
        with self.lock:
            self.state['message'] = message

    def worker(self):
        engine = None
        try:
            from web_embedding_engine import Engine
            from PIL import Image, UnidentifiedImageError
            Image.MAX_IMAGE_PIXELS = 20_000_000
            engine = Engine(self.update_state)
            with self.lock:
                self.state.update(status='ready', message='模型就绪', model_load_seconds=engine.load_seconds)
            while True:
                job_id, data = self.pending.get()
                with self.lock:
                    self.state['active_job'] = job_id
                    self.jobs[job_id].update(status='running', started_at=time.time())
                def progress(message):
                    with self.lock:
                        self.jobs[job_id]['progress'] = message
                try:
                    image = None
                    if data['image_bytes'] is not None:
                        try:
                            image = Image.open(io.BytesIO(data['image_bytes']))
                            if image.width * image.height > 20_000_000:
                                raise ValueError('图片分辨率过大，请先缩小至2000万像素以内。')
                            image.load()
                            image = image.convert('RGB')
                        except (UnidentifiedImageError, OSError) as error:
                            raise ValueError('无法读取图片，请使用PNG或JPEG。') from error
                    result = engine.compare(data['text'], image, data['candidates'], progress)
                    with self.lock:
                        self.jobs[job_id].update(status='complete', result=result, finished_at=time.time())
                    print(json.dumps({'job': job_id, 'status': 'complete', 'kind': result['kind'],
                                      'total_ms': result['total_ms']}, ensure_ascii=False), flush=True)
                except Exception as error:
                    message = str(error) if isinstance(error, ValueError) else '推理失败，请检查板端服务日志。'
                    with self.lock:
                        self.jobs[job_id].update(status='failed', error=message, finished_at=time.time())
                    print(repr(error), flush=True)
                finally:
                    with self.lock:
                        self.state['active_job'] = None
                    self.pending.task_done()
        except Exception as error:
            with self.lock:
                self.state.update(status='failed', message='模型启动失败，请检查板端日志')
            print(repr(error), flush=True)
        finally:
            if engine is not None:
                engine.close()

    def submit(self, data):
        with self.lock:
            if self.state['status'] != 'ready':
                return 503, {'error': '模型尚未就绪，请稍后再试。'}
            if self.pending.full():
                return 429, {'error': '请求队列已满，请等当前测试完成。'}
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {'id': job_id, 'status': 'queued', 'progress': '等待BPU执行', 'created_at': time.time()}
            self.pending.put_nowait((job_id, data))
            for old in list(self.jobs):
                if len(self.jobs) <= 80:
                    break
                if self.jobs[old]['status'] in ('complete', 'failed'):
                    del self.jobs[old]
            return 202, {'id': job_id}


class Handler(BaseHTTPRequestHandler):
    server_version = 'EmbeddingLab/1.0'
    def send(self, code, data, content_type='application/json; charset=utf-8'):
        body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, dict) else data
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        service = self.server.service
        if path == '/api/status':
            with service.lock:
                self.send(200, dict(service.state, queued=service.pending.qsize()))
        elif path.startswith('/api/jobs/'):
            job_id = path.rsplit('/', 1)[-1]
            with service.lock:
                job = service.jobs.get(job_id)
                self.send(200 if job else 404, job or {'error': '测试记录已过期，请重新测试。'})
        elif path.startswith('/samples/') and path.rsplit('/', 1)[-1] in SAMPLES:
            self.send(200, (ROOT / SAMPLES[path.rsplit('/', 1)[-1]]).read_bytes(), 'image/png' if path.endswith(('bridge','cat')) else 'image/jpeg')
        elif path in ('/', '/app.js', '/style.css'):
            name = 'index.html' if path == '/' else path[1:]
            mime = {'index.html': 'text/html; charset=utf-8', 'app.js': 'application/javascript; charset=utf-8', 'style.css': 'text/css; charset=utf-8'}[name]
            self.send(200, (STATIC / name).read_bytes(), mime)
        else:
            self.send(404, {'error': '未找到页面。'})

    def do_POST(self):
        if self.path != '/api/jobs':
            return self.send(404, {'error': '未找到接口。'})
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.send(403, {'error': '只接受本页面的请求。'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                return self.send(413, {'error': '请求过大，图片最大5MB。'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send(415, {'error': '需要JSON请求。'})
            data = validate(json.loads(self.rfile.read(length)))
            code, result = self.server.service.submit(data)
            self.send(code, result)
        except (ValueError, UnicodeDecodeError) as error:
            self.send(400, {'error': str(error)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8088, type=int)
    args = parser.parse_args()
    service = Service()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.service = service
    print('LAN UI listening on %s:%s' % (args.host, args.port), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
