"""Board text/image pipeline validation with persistent compiled cores.

Timing includes CPU preprocessing, embedding lookup, copies, vision/fusion
runtime and normalization. Model loading is recorded separately. Two image
examples are a prototype check, never independent production acceptance.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from benchmark import state
from persistent_bpu import Runtime
from prepare_multimodal_inputs import prepare,fuse

parser=argparse.ArgumentParser()
parser.add_argument('--repeats',type=int,default=10)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=root/'bpu/results/multimodal_bpu_runtime'
out.mkdir(exist_ok=True)
report={'status':'running','started_at':time.time(),'device':'RDK X5 CPU+BPU',
        'note':'Prototype text and two-image checks; audio/video and independent production dataset not accepted',
        'state_before':state(),'software_checks':[],'pipeline_cases':[]}

def save():
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

def normalized(raw):
    vector=raw.reshape(768)
    assert np.isfinite(vector).all() and np.linalg.norm(vector)>0
    return vector/np.linalg.norm(vector)

vision=fusion=None
try:
    save()
    manifest=json.loads((root/'bpu/runtime/multimodal_deployment.json').read_text())
    for item in manifest['models']:
        path=root/item['relative_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], 'Deployment hash mismatch'
    report['deployment']=manifest
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    start=time.perf_counter()
    model=SentenceTransformer(str(root/'model'),device='cpu',local_files_only=True,
        config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'dtype':torch.float32}).eval()
    vision=Runtime(root/'bpu/models/vision_a16_w8_sm8/embeddinggemma2_vision_s70.bin',[(1,768,1,630)])
    fusion=Runtime(root/'bpu/fusion_l128/a16_w8_sm8/embeddinggemma2_fusion_l128.bin',[(1,512,1,128),(1,1,1,128)])
    assert vision.output_shape==(1,512,1,64) and fusion.output_shape==(1,768,1,1)
    report['load_seconds']=time.perf_counter()-start
    source=json.loads((root/'bpu/fusion_l128/export_validation.json').read_text())
    refs=np.load(root/'bpu/fusion_l128/reference_embeddings.npy')
    software=np.load(root/'bpu/fusion_l128/a16_w8_sm8/predictions.npy')
    predictions=[]
    for definition in source['checks']:
        index=definition['index'];folder=root/'bpu/fusion_l128'/definition['group']
        arrays=[np.fromfile(folder/f'{name}/{index:03d}.bin',np.float32).reshape(shape)
                for name,shape in zip(['inputs_embeds','attention_mask'],fusion.shapes)]
        raw,ms=fusion.run(arrays);pred=normalized(raw);predictions.append(pred)
        row={'case':definition['case'],'index':index,'cosine':float(pred@refs[index]),
             'software_max_abs_error':float(np.max(np.abs(pred-software[index]))),'core_ms':ms}
        report['software_checks'].append(row);save()
    predictions=np.asarray(predictions);np.save(out/'fusion_sample_predictions.npy',predictions)
    for group in ['text_calibration','text_heldout']:
        indices=[d['index'] for d in source['checks'] if d['case'].startswith(group)]
        values=predictions[indices]
        ranking=np.argmax(values[8:]@values[:8].T,axis=1)
        report[group]={'top1':int(np.sum(ranking==np.arange(8))),'total':8,'indices':ranking.tolist()}
    semantic=root/'bpu/results/multimodal_semantic_diagnosis'
    captions=json.loads((semantic/'summary.json').read_text())['captions']
    caption_embeddings=[]
    for caption in captions:
        p=prepare(model,caption)
        raw,_=fusion.run(fuse(p));caption_embeddings.append(normalized(raw))
    caption_embeddings=np.asarray(caption_embeddings)
    np.save(out/'caption_embeddings.npy',caption_embeddings)
    reference_positions=None
    core_inputs={}
    for i,name in enumerate(['GoldenGate.png','kitchen_painting.jpg']):
        calibrated=np.fromfile(root/f'bpu/calibration/vision_pixels/{i:03d}.bin',np.float32).reshape(1,768,1,630)
        for mixed in [False,True]:
            label=f'{i:03d}_'+('mixed' if mixed else 'image')
            reference=np.load(semantic/f'{label}_embeddings.npz')['original']
            times=[];stages=[];results=[]
            for repeat in range(args.repeats+1):
                start=time.perf_counter()
                value={'image':Image.open(root/'samples'/name).convert('RGB')}
                if mixed:value['text']=captions[i]+' <|image|>'
                p=prepare(model,value)
                if reference_positions is None:reference_positions=p['positions']
                assert torch.equal(reference_positions,p['positions']), 'Image grid mismatch'
                raw,vision_ms=vision.run([p['pixels']])
                features=raw.squeeze(2).transpose(0,2,1)[0]
                arrays=fuse(p,features)
                raw,fusion_ms=fusion.run(arrays)
                pred=normalized(raw)
                scores=pred@caption_embeddings.T
                elapsed=(time.perf_counter()-start)*1000
                assert np.array_equal(calibrated,p['pixels']), 'Board preprocessing mismatch'
                results.append({'repeat':repeat,'cosine':float(pred@reference),
                    'top1':int(np.argmax(scores)),'top1_correct':int(np.argmax(scores)) in [i,i+4],
                    'scores':scores.tolist()})
                times.append(elapsed);stages.append({'vision_core_ms':vision_ms,'fusion_core_ms':fusion_ms})
                np.save(out/f'{label}_repeat_{repeat:03d}.npy',pred)
                core_inputs={'vision':[p['pixels']],'fusion':arrays}
            row={'case':label,'image':name,'mixed':mixed,'tokens':p['tokens'],
                 'cold_first_call_ms':times[0],'raw_warm_ms':times[1:],
                 'median_end_to_end_ms':float(np.median(times[1:])),
                 'p95_end_to_end_ms':float(np.percentile(times[1:],95)),
                 'stages':stages,'results':results,'state_after':state()}
            report['pipeline_cases'].append(row);save()
    for scope,runtime in [('vision',vision),('fusion',fusion)]:
        measurements=[]
        for frame in range(100):
            start=time.perf_counter();_,ms=runtime.run(core_inputs[scope])
            measurements.append({'frame':frame,'infer_wait_ms':ms,'call_ms':(time.perf_counter()-start)*1000})
        report[scope+'_100_frames']={'measurements':measurements,
            'mean_core_ms':float(np.mean([m['infer_wait_ms'] for m in measurements])),
            'mean_call_ms':float(np.mean([m['call_ms'] for m in measurements])),'state_after':state()}
        save()
    report['minimum_fusion_cosine']=min(r['cosine'] for r in report['software_checks'])
    report['minimum_joint_cosine']=min(r['cosine'] for case in report['pipeline_cases'] for r in case['results'])
    report['prototype_accuracy_gate_passed']=report['minimum_fusion_cosine']>=.95 and report['minimum_joint_cosine']>=.95 and all(
        r['top1_correct'] for case in report['pipeline_cases'] for r in case['results']) and all(
        report[group]['top1']==8 for group in ['text_calibration','text_heldout'])
    report.update(status='complete',finished_at=time.time(),state_end=state())
    save()
except Exception as error:
    report.update(status='failed',error=repr(error),finished_at=time.time());save()
    raise
finally:
    if fusion:fusion.close()
    if vision:vision.close()
