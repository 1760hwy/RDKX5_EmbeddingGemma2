"""Make padding-mask dynamic range usable for INT8 and verify exact test inputs.

This changes only exported mask constants, not model weights. Numerical
equivalence is checked on every current reference input, not assumed globally.
"""
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

root=Path('/work/bpu')
reports={}
for scope,source in [('text','text_core_l32'),('vision','vision_core_s70')]:
    graph=onnx.load(root/'models'/f'{source}.onnx')
    changes=[]
    for tensor in graph.graph.initializer:
        array=numpy_helper.to_array(tensor)
        if array.dtype==np.float32 and np.any(array==-1e9):
            # Only the mask initializer may carry this sentinel.
            assert 'mask' in tensor.name,tensor.name
            changed=np.where(array==-1e9,np.float32(-32),array)
            tensor.CopyFrom(numpy_helper.from_array(changed,tensor.name))
            changes.append(tensor.name)
    for node in graph.graph.node:
        if node.op_type!='Constant': continue
        for attr in node.attribute:
            if attr.name!='value': continue
            array=numpy_helper.to_array(attr.t)
            if array.dtype==np.float32 and np.any(array==-1e9):
                assert array.size==1,(node.name,array.shape)
                attr.t.CopyFrom(numpy_helper.from_array(np.full_like(array,-32)))
                changes.append(node.name)
    assert len(changes)==1,(scope,changes)
    target=root/'models'/f'{source}_bounded_mask.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph,target)
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    refs=np.load(root/'results'/('reference_embeddings.npy' if scope=='text' else 'vision_reference_features.npy'))
    checks=[]
    for i,ref in enumerate(refs):
        if scope=='text':
            inputs={'inputs_embeds':np.fromfile(root/f'calibration/inputs_embeds/{i:03d}.bin',np.float32).reshape(1,512,1,32),
                'attention_mask':np.fromfile(root/f'calibration/attention_mask/{i:03d}.bin',np.float32).reshape(1,1,1,32)}
            out=session.run(None,inputs)[0].reshape(768)
        else:
            inputs={'pixel_values':np.fromfile(root/f'calibration/vision_pixels/{i:03d}.bin',np.float32).reshape(1,768,1,630)}
            out=session.run(None,inputs)[0].squeeze(2).transpose(0,2,1)[0]
        cosine=float(out.ravel()@ref.ravel()/(np.linalg.norm(out)*np.linalg.norm(ref)))
        error=float(np.max(np.abs(out-ref)))
        relative=float(np.linalg.norm(out-ref)/np.linalg.norm(ref))
        assert cosine>0.99999 and error<(1e-4 if scope=='text' else 5e-4) and relative<1e-4,(scope,i,cosine,error,relative)
        checks.append({'case':i,'cosine':cosine,'max_abs_error':error,'relative_rmse':relative})
        print('BOUNDED_MASK_VERIFIED',scope,checks[-1],flush=True)
    report={'status':'complete','scope':scope,'mask_value':-32,'changed_constants':changes,
        'model':str(target),'checks':checks,
        'limitation':'Numerical equivalence checked on current examples; independent data must verify bounded-mask effects'}
    (root/'results'/f'{scope}_bounded_mask_validation.json').write_text(json.dumps(report,indent=2))
    reports[scope]=report
print('ALL_BOUNDED_MASKS_VERIFIED',flush=True)
