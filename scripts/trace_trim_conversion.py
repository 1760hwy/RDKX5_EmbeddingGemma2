"""Read-only intermediate direction comparisons; no model correction."""
import gc,json,time
from pathlib import Path
import numpy as np
from horizon_nn.ir import OnnxModel
from horizon_nn.executor import ORTExecutor
root=Path('/work/bpu/vision_trim576')
folder=root/'a16_w8_sm8'
prefix='embeddinggemma2_vision_trim576'
destination=folder/'conversion_trace.json'
report={'status':'running','started_at':time.time(),
        'scope':'one image, fake calibration versus converted integer tensor directions',
        'note':'Scalar feature scales cancel in cosine; no magnitude/board acceptance claim'}
def save():destination.write_text(json.dumps(report,indent=2))
save()
try:
    fake=OnnxModel(str(folder/f'{prefix}_mixed_calibrated_model.onnx'))
    quant=OnnxModel(str(folder/f'{prefix}_quantized_model.onnx'))
    variables={v.name:v for n in quant.graph.nodes for v in n.outputs}
    pairs=[]
    for n in fake.graph.type2nodes['HzCalibration']:
        if n.tensor_type!='feature' or len(n.thresholds)!=1:continue
        first='/layers.0/' in n.name
        residual=any(f'/layers.{i}/Add_1_output_0_' in n.name for i in range(1,16))
        if not (first or residual):continue
        variable=n.outputs[0]
        candidate=variable.name.replace('_calibrated','_quantized')
        if candidate not in variables:continue
        q=variables[candidate]
        if str(q.dtype) not in ['DataType.INT8','DataType.INT16']:continue
        pairs.append((n.name,variable.name,candidate))
        fake.graph.append_output(variable);quant.graph.append_output(q)
    assert pairs
    report['matched_pairs']=pairs;save()
    x=np.fromfile(root/'calibration/vision_pixels/000.bin',np.float32).reshape(1,768,1,576)
    executor=ORTExecutor(fake).to('cpu').create_session()
    a=executor.forward({'pixel_values':x},[p[1] for p in pairs])
    del executor,fake;gc.collect()
    executor=ORTExecutor(quant).to('cpu').create_session()
    b=executor.forward({'pixel_values':x},[p[2] for p in pairs])
    rows=[]
    for node,fa,qb in pairs:
        left=a[fa];right=b[qb].astype(np.float32)
        if left.shape!=right.shape:
            if left.ndim==4 and left.transpose(0,2,3,1).shape==right.shape:
                left=left.transpose(0,2,3,1)
            elif right.ndim==4 and right.transpose(0,2,3,1).shape==left.shape:
                right=right.transpose(0,2,3,1)
            else:
                rows.append({'node':node,'status':'unmatched_layout','fake_shape':list(left.shape),'quant_shape':list(right.shape)})
                continue
        cosine=float(left.ravel()@right.ravel()/(np.linalg.norm(left)*np.linalg.norm(right)))
        row={'node':node,'shape':list(left.shape),'direction_cosine':cosine,
             'quant_zero_fraction':float(np.mean(right==0))}
        rows.append(row);print(row,flush=True)
    report.update(status='complete',checks=rows,finished_at=time.time());save()
except Exception as error:
    report.update(status='failed',error=repr(error));save();raise
