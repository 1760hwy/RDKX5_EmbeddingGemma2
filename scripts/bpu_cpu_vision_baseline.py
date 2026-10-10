"""Board baseline matching the 70-soft-token exported visual core."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
root=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
start=time.perf_counter()
model=SentenceTransformer(str(root/'model'),device='cpu',local_files_only=True,
    config_kwargs={'audio_config':None},
    model_kwargs={'dtype':torch.float32,'attn_implementation':'eager'})
model.eval()
load=time.perf_counter()-start
names=['GoldenGate.png','kitchen_painting.jpg']
features=[model[0].preprocess([{'image':Image.open(root/'samples'/name).convert('RGB')}],
    processing_kwargs={'image':{'max_soft_tokens':70}}) for name in names]
ref=np.load(root/'bpu/results/vision_reference_features.npy')
def forward(i):
    f=features[i]
    return model[0].auto_model.get_image_features(f['pixel_values'],f['image_position_ids']).pooler_output[0].numpy()
with torch.inference_mode():
    checks=[]
    for i in range(2):
        result=forward(i)
        cosine=float(result.ravel()@ref[i].ravel()/(np.linalg.norm(result)*np.linalg.norm(ref[i])))
        error=float(np.max(np.abs(result-ref[i])))
        assert cosine>0.99999 and error<5e-4,(i,cosine,error)
        checks.append({'sample':names[i],'cosine':cosine,'max_abs_error':error})
        print('REFERENCE_VALIDATED',checks[-1],flush=True)
    forward(0)
    raw=[]
    for _ in range(5):
        start=time.perf_counter()
        forward(0)
        raw.append(time.perf_counter()-start)
        print('VISION_CORE_SECONDS',raw[-1],flush=True)
report={'status':'complete','threads':4,'device':'X5 CPU','dtype':'float32',
    'scope':'original visual core and projection; excludes preprocessing, text network and final embedding',
    'max_soft_tokens':70,'load_seconds':load,'raw_seconds':raw,
    'median_ms':float(np.median(raw)*1000),'p95_ms':float(np.percentile(raw,95)*1000),
    'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000,'checks':checks}
(root/'bpu/results/cpu_vision_core_s70.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
