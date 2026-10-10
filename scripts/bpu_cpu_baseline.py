"""Board CPU baseline for exactly the fixed-length core exported to ONNX."""
import json
import os
import resource
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
t=time.perf_counter()
model=SentenceTransformer(str(ROOT/'model'),device='cpu',local_files_only=True,
    config_kwargs={'audio_config':None,'vision_config':None},
    model_kwargs={'dtype':torch.float32,'attn_implementation':'eager'})
model.eval()
load_seconds=time.perf_counter()-t
network=model[0].auto_model.language_model
ref=np.load(ROOT/'bpu/results/reference_embeddings.npy')
cases=[]
outputs=[]
prepared=[]
for i in range(16):
    x=np.fromfile(ROOT/f'bpu/calibration/inputs_embeds/{i:03d}.bin',dtype=np.float32).reshape(1,512,1,32)
    m=np.fromfile(ROOT/f'bpu/calibration/attention_mask/{i:03d}.bin',dtype=np.float32).reshape(1,32)
    prepared.append((torch.from_numpy(x).squeeze(2).transpose(1,2),torch.from_numpy(m)))
def forward(index):
    x,valid=prepared[index]
    additive=(1-valid[:,None,None,:])*-1e9
    h=network(inputs_embeds=x,attention_mask={'full_attention':additive,'sliding_attention':additive}).last_hidden_state
    mean=(h*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True).clamp(min=1)
    return torch.nn.functional.normalize(mean,p=2,dim=1).numpy()[0]
with torch.inference_mode():
    for i in range(16):
        output=forward(i)
        cosine=float(output@ref[i])
        error=float(np.max(np.abs(output-ref[i])))
        assert cosine>0.99999 and error<1e-4,(i,cosine,error)
        outputs.append(output)
    forward(8)
    forward(8)
    samples=[]
    for _ in range(7):
        t=time.perf_counter()
        forward(8)
        samples.append(time.perf_counter()-t)
result={'status':'complete','threads':4,'device':'X5 CPU','dtype':'float32',
    'scope':'32-token text core; excludes token lookup, tokenizer, disk reads and model loading',
    'load_seconds':load_seconds,'raw_seconds':samples,'median_ms':float(np.median(samples)*1000),
    'p95_ms':float(np.percentile(samples,95)*1000),'peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000,
    'reference_cosines':[float(out@r) for out,r in zip(outputs,ref)]}
(ROOT/'bpu/results/cpu_text_core_l32.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
