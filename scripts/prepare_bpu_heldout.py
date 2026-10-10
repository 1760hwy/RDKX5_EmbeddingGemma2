"""Prepare held-out bilingual retrieval examples, never used for PTQ calibration."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer
root=Path(__file__).resolve().parents[1]
out=root/'bpu/evaluation'
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
docs=['按下急停按钮后，机器人停止运动。','摄像头曝光过高，图像发白。',
    '电机过热时，需要降低负载。','温度传感器通过串口发送数据。',
    '模型量化后在BPU上进行推理。','使用SSH远程登录开发板。',
    '数据库保存设备的故障日志。','语音识别将声音转换为文字。']
queries=['How can I immediately stop the robot?', 'Why is the camera image overexposed?',
    'What should I do when the motor overheats?', 'How does the temperature sensor send readings?',
    'Where does the quantized model run?', 'How do I remotely access the development board?',
    'Where are equipment fault records stored?', 'How can speech be converted into text?']
model=SentenceTransformer(str(root/'model'),device='cpu',local_files_only=True,
    config_kwargs={'audio_config':None,'vision_config':None},
    model_kwargs={'dtype':torch.float32,'attn_implementation':'eager'})
model.max_seq_length=32
model.eval()
options=ort.SessionOptions()
options.intra_op_num_threads=4
options.inter_op_num_threads=1
session=ort.InferenceSession(str(root/'bpu/models/text_core_l32_bounded_mask.onnx'),options,
    providers=['CPUExecutionProvider'])
refs=[]
cases=[]
for i,(text,prompt) in enumerate([(x,'Document') for x in docs]+[(x,'SearchQuery') for x in queries]):
    f=model.tokenize([model.prompts[prompt]+text])
    ids,mask=f['input_ids'],f['attention_mask']
    padding=32-ids.shape[1]
    ids=torch.nn.functional.pad(ids,(0,padding),value=0)
    mask=torch.nn.functional.pad(mask,(0,padding),value=0)
    with torch.no_grad():
        x=model[0].auto_model.get_input_embeddings()(ids).transpose(1,2).unsqueeze(2).numpy()
        valid=mask.float().reshape(1,1,1,32).numpy()
        ref=model.encode([text],prompt_name=prompt,normalize_embeddings=True,show_progress_bar=False)[0]
    result=session.run(None,{'inputs_embeds':x,'attention_mask':valid})[0].reshape(768)
    cosine=float(result@ref)
    error=float(np.abs(result-ref).max())
    assert cosine>0.99999 and error<1e-4,(i,cosine,error)
    for name,array in [('inputs_embeds',x),('attention_mask',valid)]:
        folder=out/name
        folder.mkdir(parents=True,exist_ok=True)
        array.astype(np.float32).tofile(folder/f'{i:03d}.bin')
    refs.append(ref)
    cases.append({'text':text,'prompt':prompt,'tokens':int(mask.sum()),'onnx_cosine':cosine,'max_abs_error':error})
    print('HELDOUT_FLOAT_VALIDATED',i,cosine,error,flush=True)
refs=np.asarray(refs)
ranking=np.argmax(refs[8:]@refs[:8].T,axis=1)
np.save(out/'reference_embeddings.npy',refs)
report={'status':'complete','cases':cases,'float_retrieval_top1':int(np.sum(ranking==np.arange(8))),
    'retrieval_total':8,'float_ranking':ranking.tolist(),
    'note':'Authored held-out examples excluded from calibration; not a production dataset'}
(out/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
