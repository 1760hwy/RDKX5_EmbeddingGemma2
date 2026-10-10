"""Isolated padding-only vision candidate; evaluate against original semantics.

No weights are pruned. Softmax stays INT8. Compilation is a separate gate.
"""
import hashlib
import importlib
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import yaml
from horizon_nn.ir import OnnxModel, save_model
from horizon_nn.builder.model_builder import ModelBuilder
from horizon_nn.executor import ORTExecutor

ROOT = Path('/work/bpu/vision_trim576')
fresh_builder = '--fresh-builder' in sys.argv
OUT = ROOT/('a16_w8_sm8_fresh' if fresh_builder else 'a16_w8_sm8')
OUT.mkdir(exist_ok=True)
report = {'status':'running','started_at':time.time(),'checks':[],
          'weights_pruned':False,'board_tested':False,'fresh_conversion_builder':fresh_builder}
logging.basicConfig(level=logging.INFO)

def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    exported=json.loads((ROOT/'results/vision_export_validation.json').read_text())
    assert exported['status']=='complete' and exported['trim_padding']
    assert exported['input_shape']==[1,768,1,576]
    graph=ROOT/'models/vision_core_s70.onnx'
    identity=digest(graph)
    provenance=OUT/'source_sha256.txt'
    if provenance.exists():
        assert provenance.read_text().strip()==identity
    else:
        provenance.write_text(identity)
    report['source_sha256']=identity
    conf=yaml.safe_load(Path('/work/bpu/vision_core_s70.yaml').read_text())
    conf['model_parameters'].update(onnx_model=str(graph),working_dir=str(OUT),
        output_model_file_prefix='embeddinggemma2_vision_trim576')
    conf['input_parameters']['input_shape']='1x768x1x576'
    conf['calibration_parameters']['cal_data_dir']=str(ROOT/'calibration/vision_pixels')
    config=OUT/'config.yaml'
    config.write_text(yaml.safe_dump(conf,sort_keys=False))
    mapper=importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
    runner=mapper.MakertbinRunner(str(config),'onnx');runner._parse_conf()
    builder=ModelBuilder(OnnxModel(str(graph)),march='bayes-e',
        optimization=['set_Softmax_input_int8'],input_dict=runner._get_build_input_params(),
        cali_dict=runner._get_build_calibration_params())
    name='embeddinggemma2_vision_trim576'
    calibrated=OUT/f'{name}_calibrated_model.onnx'
    quantized=OUT/f'{name}_quantized_model.onnx'
    if fresh_builder:
        # The established 630-patch candidate used a fresh conversion builder.
        # Reuse only the completed calibration for this exact 576-patch export.
        previous=ROOT/'a16_w8_sm8'
        assert (previous/'source_sha256.txt').read_text().strip()==identity
        calibrated=previous/f'{name}_calibrated_model.onnx'
        assert calibrated.is_file()
        report['calibrated_sha256']=digest(calibrated)
        report['calibration_source']=str(calibrated)
    if calibrated.exists():
        model=OnnxModel(str(calibrated));report['calibration_reused']=True
    else:
        report['phase']='prepare_optimize_calibrate';save()
        original=builder.prepare(builder.input_model)
        save_model(original,str(OUT/f'{name}_original_float_model.onnx'))
        optimized=builder.optimize(original)
        save_model(optimized,str(OUT/f'{name}_optimized_float_model.onnx'))
        model,_,_=builder.calibrate(optimized)
        save_model(model,str(calibrated))
    protected=[]
    for node in model.graph.type2nodes['HzCalibration']:
        softmax='Softmax' in node.name or any('FROM_QUANTIZED_SOFTMAX' in n.name for n in node.next_ops)
        if node.tensor_type=='feature' and softmax:
            assert node.qtype=='int8';protected.append(node.name)
        elif node.tensor_type=='feature':
            node.qtype='int16'
    assert protected
    report['protected_nodes']=protected
    if quantized.exists():
        quant=OnnxModel(str(quantized));report['quantization_reused']=True
    else:
        report['phase']='vendor_quantization';save()
        save_model(model,str(OUT/f'{name}_mixed_calibrated_model.onnx'))
        converter=ModelBuilder(model,march='bayes-e',optimization=['set_Softmax_input_int8']) if fresh_builder else builder
        quant=converter.quantize(model);save_model(quant,str(quantized))
    report['quantized_sha256']=digest(quantized)
    report['phase']='joint_software_validation';save()
    vision=ORTExecutor(quant).to('cpu').create_session()
    refs=np.load(ROOT/'results/vision_reference_features.npy')
    predictions=[]
    for i,ref in enumerate(refs):
        x=np.fromfile(ROOT/f'calibration/vision_pixels/{i:03d}.bin',np.float32).reshape(1,768,1,576)
        raw=next(iter(vision.forward({'pixel_values':x}).values()))
        assert raw.shape==(1,512,1,64) and np.isfinite(raw).all()
        pred=raw.squeeze(2).transpose(0,2,1)[0];predictions.append(pred)
        report['checks'].append({'image_index':i,'intermediate_cosine':float(
            pred.ravel()@ref.ravel()/(np.linalg.norm(pred)*np.linalg.norm(ref)))})
        save()
    np.save(OUT/'calibration_predictions.npy',np.asarray(predictions))
    del vision
    fusion_path=Path('/work/bpu/fusion_l128/a16_w8_sm8/embeddinggemma2_fusion_l128_quantized_model.onnx')
    expected='34f48090551d69fd7b42050a91412bc7201a634fa0492d8afe5ee91880fe9a3e'
    assert digest(fusion_path)==expected
    report['fusion_quantized_sha256']=expected
    fusion=ORTExecutor(str(fusion_path)).to('cpu').create_session()
    semantic=Path('/work/bpu/results/multimodal_semantic_diagnosis')
    captions=np.load(semantic/'caption_embeddings.npy')
    joint=[]
    for i,pred in enumerate(predictions):
        for kind in ['image','mixed']:
            case=f'{i:03d}_{kind}'
            inputs=np.load(semantic/f'{case}_fusion_inputs.npz')
            x=inputs['inputs_embeds'].copy()
            # Existing reference inputs contain the exact 64 image slots.
            original=np.load(Path('/work/bpu/results/vision_reference_features.npy'))[i]
            matches=np.max(np.abs(x[0,:,None,:]-original[None,:,:]),axis=2)<1e-5
            assert np.all(matches.sum(axis=0)==1), 'Ambiguous reference image slot'
            indices=np.argmax(matches,axis=0)
            assert len(np.unique(indices))==64
            x[0,indices]=pred
            n=x.shape[1]
            padded=np.pad(x,((0,0),(0,128-n),(0,0))).transpose(0,2,1)[:,:,None,:]
            mask=np.pad(inputs['attention_mask'],((0,0),(0,128-n)))[:,None,None,:]
            raw=next(iter(fusion.forward({'inputs_embeds':padded.astype(np.float32),
                'attention_mask':mask.astype(np.float32)}).values())).reshape(768)
            assert np.isfinite(raw).all() and np.linalg.norm(raw)>0
            output=raw/np.linalg.norm(raw)
            reference=np.load(semantic/f'{case}_embeddings.npz')['original']
            rankings=np.argsort(-(output@captions.T))
            row={'case':case,'cosine':float(output@reference),
                 'caption_rankings':rankings.tolist(),'top1_correct':int(rankings[0]) in [i,i+4]}
            joint.append(row);print(row,flush=True)
    report['joint_checks']=joint
    report['minimum_joint_cosine']=min(row['cosine'] for row in joint)
    report['joint_gate_passed']=all(row['cosine']>=.95 and row['top1_correct'] for row in joint)
    report.update(status='complete',finished_at=time.time());save()

if __name__=='__main__':
    try:
        save();main()
    except Exception as error:
        report.update(status='failed',error=repr(error),finished_at=time.time());save()
        raise
