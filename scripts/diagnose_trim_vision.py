"""Separate export/optimizer/fake-quant/conversion error on one fixed input."""
import gc
import json
import time
from pathlib import Path
import numpy as np
from horizon_nn.ir import OnnxModel
from horizon_nn.executor import ORTExecutor

root=Path('/work/bpu/vision_trim576')
folder=root/'a16_w8_sm8'
prefix='embeddinggemma2_vision_trim576'
report={'status':'running','scope':'one calibration image, software diagnosis only','stages':[]}
destination=folder/'stage_diagnosis.json'
def save():
    destination.write_text(json.dumps(report,indent=2))
save()
try:
    refs=np.load(root/'results/vision_reference_features.npy')[0]
    inputs={'pixel_values':np.fromfile(root/'calibration/vision_pixels/000.bin',np.float32).reshape(1,768,1,576)}
    for stage in ['optimized_float','calibration_off','mixed_fake_quant','quantized']:
        suffix={'calibration_off':'calibrated','mixed_fake_quant':'mixed_calibrated'}.get(stage,stage)
        model=OnnxModel(str(folder/f'{prefix}_{suffix}_model.onnx'))
        if stage=='calibration_off':
            for node in model.graph.type2nodes['HzCalibration']:
                node.switch='OFF'
        executor=ORTExecutor(model).to('cpu').create_session()
        start=time.time()
        raw=next(iter(executor.forward(inputs).values()))
        pred=raw.squeeze(2).transpose(0,2,1)[0]
        assert pred.shape==refs.shape and np.isfinite(pred).all()
        row={'stage':stage,'cosine':float(pred.ravel()@refs.ravel()/(np.linalg.norm(pred)*np.linalg.norm(refs))),
             'relative_rmse':float(np.linalg.norm(pred-refs)/np.linalg.norm(refs)),
             'seconds':time.time()-start}
        report['stages'].append(row);save();print(row,flush=True)
        del executor,model;gc.collect()
    old=OnnxModel('/work/bpu/models/vision_a16_w8_sm8/embeddinggemma2_vision_s70_calibrated_model.onnx')
    new=OnnxModel(str(folder/f'{prefix}_mixed_calibrated_model.onnx'))
    def parameters(model):
        return {node.name:{'qtype':node.qtype,'thresholds':list(node.thresholds) if node.thresholds else [],
                          'tensor_type':node.tensor_type}
                for node in model.graph.type2nodes['HzCalibration'] if node.tensor_type=='feature'}
    report['old_parameters']=parameters(old);report['new_parameters']=parameters(new)
    report.update(status='complete');save()
except Exception as error:
    report.update(status='failed',error=repr(error));save();raise
