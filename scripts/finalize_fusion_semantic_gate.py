"""Check accumulated vision/fusion error from saved quantized predictions."""
import json
from pathlib import Path
import numpy as np

root=Path('/work/bpu')
folder=root/'fusion_l128/a16_w8_sm8'
report=json.loads((folder/'validation.json').read_text())
assert report['status']=='complete'
source=json.loads((root/'fusion_l128/export_validation.json').read_text())
predictions=np.load(folder/'predictions.npy')
semantic=root/'results/multimodal_semantic_diagnosis'
captions=np.load(semantic/'caption_embeddings.npy')
joint=[]
for row, definition in zip(report['checks'],source['checks']):
    assert row['case']==definition['case'] and row['index']==definition['index']
    if row['case'].endswith('_quantized_vision'):
        base=row['case'].removesuffix('_quantized_vision')
        ref=np.load(semantic/f'{base}_embeddings.npz')['original']
        pred=predictions[row['index']]
        scores=pred@captions.T
        image_index=int(base[:3])
        row.update(joint_embedding_cosine=float(pred@ref),caption_scores=scores.tolist(),
                   caption_rankings=np.argsort(-scores).tolist(),
                   top1_correct=int(np.argmax(scores)) in [image_index,image_index+4])
        joint.append(row)
assert len(joint)==4
report['minimum_joint_multimodal_cosine']=min(row['joint_embedding_cosine'] for row in joint)
report['joint_multimodal_gate_passed']=all(row['joint_embedding_cosine']>=.95 and row['top1_correct'] for row in joint)
report['software_gate_passed']=report['minimum_cosine']>=.95 and all(
    report[group+'_top1']==8 for group in ['text_calibration','text_heldout']) and report['joint_multimodal_gate_passed']
(folder/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({key:report[key] for key in ['minimum_cosine','minimum_joint_multimodal_cosine','software_gate_passed']},indent=2))
