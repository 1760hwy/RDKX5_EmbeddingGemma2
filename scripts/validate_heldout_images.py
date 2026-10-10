"""Validate the exact fresh vision/fusion pair on uncalibrated cat images."""
import hashlib,json,time,gc
from pathlib import Path
import numpy as np
from horizon_nn.executor import ORTExecutor
root=Path('/work/bpu')
folder=root/'vision_trim576/heldout_images_letterbox'
report={'status':'waiting_references','started_at':time.time(),'cases':[],
        'board_tested':False,'scope':'four correlated frames, two input modes each; square letterbox'}
destination=folder/'software_validation.json'
def save():destination.write_text(json.dumps(report,indent=2))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
try:
    save();deadline=time.monotonic()+1200
    ready=root/'logs/multimodal_repair/heldout_letterbox_prepare.exit'
    while not ready.exists():
        assert time.monotonic()<deadline, 'Reference preparation wait timed out'
        time.sleep(15)
    assert ready.read_text().strip()=='0', 'Reference preparation failed; inspect original log'
    source=json.loads((folder/'references.json').read_text())
    assert source['status']=='complete' and len(source['cases'])==8
    vp=root/'vision_trim576/a16_w8_sm8_fresh/embeddinggemma2_vision_trim576_quantized_model.onnx'
    fp=root/'fusion_l128/a16_w8_sm8/embeddinggemma2_fusion_l128_quantized_model.onnx'
    assert sha(vp)=='bbaeb03aa9b6f756e5990fcfafe1582c619a693a994fbce5f0dae32b981b8a12'
    assert sha(fp)=='34f48090551d69fd7b42050a91412bc7201a634fa0492d8afe5ee91880fe9a3e'
    report.update(status='running',vision_sha256=sha(vp),fusion_sha256=sha(fp));save()
    vision=ORTExecutor(str(vp)).to('cpu').create_session()
    features={}
    for i in range(1,5):
        data=np.load(folder/f'cat_frame_{i:02d}_image.npz')
        pred=next(iter(vision.forward({'pixel_values':data['pixels']}).values()))
        assert pred.shape==(1,512,1,64) and np.isfinite(pred).all()
        features[i]=pred.squeeze(2).transpose(0,2,1)[0]
        np.save(folder/f'cat_frame_{i:02d}_vision_quantized.npy',features[i])
        print('VISION_DONE',i,flush=True)
    del vision;gc.collect()
    fusion=ORTExecutor(str(fp)).to('cpu').create_session()
    captions=np.load(root/'results/multimodal_semantic_diagnosis/caption_embeddings.npy')
    outputs=[]
    for case in source['cases']:
        label=case['case'];i=int(label.split('_')[2]);data=np.load(folder/f'{label}.npz')
        x=data['inputs_embeds'].copy();x[data['image_mask']]=features[i]
        n=x.shape[1]
        padded=np.pad(x,((0,0),(0,128-n),(0,0))).transpose(0,2,1)[:,:,None,:]
        mask=np.pad(data['attention_mask'],((0,0),(0,128-n)))[:,None,None,:]
        pred=next(iter(fusion.forward({'inputs_embeds':padded.astype(np.float32),
            'attention_mask':mask.astype(np.float32)}).values())).reshape(768)
        assert np.isfinite(pred).all() and np.linalg.norm(pred)>0
        pred/=np.linalg.norm(pred)
        ref=data['reference'];rank=np.argsort(-(pred@captions.T));ref_rank=np.argsort(-(ref@captions.T))
        row={'case':label,'cosine':float(pred@ref),'rankings':rank.tolist(),'reference_rankings':ref_rank.tolist(),
             'top1_correct':int(rank[0]) in case['expected_caption_indices'],
             'reference_top1_correct':int(ref_rank[0]) in case['expected_caption_indices']}
        report['cases'].append(row);outputs.append(pred);save();print(row,flush=True)
    np.save(folder/'software_predictions.npy',np.asarray(outputs))
    report['minimum_cosine']=min(r['cosine'] for r in report['cases'])
    report['heldout_gate_passed']=all(r['cosine']>=.95 and r['top1_correct'] and r['reference_top1_correct'] for r in report['cases'])
    report.update(status='complete',finished_at=time.time());save()
except Exception as error:
    report.update(status='failed',error=repr(error),finished_at=time.time());save();raise
