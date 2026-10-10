"""Vendor calibration and software validation of isolated 128-token fusion.

Checkpoints allow reuse without repeating completed calibration/conversion.
All Softmax boundaries retain INT8; all other feature requests are INT16.
"""
import hashlib
import importlib
import json
import logging
import time
from pathlib import Path
import numpy as np
import yaml
from horizon_nn.ir import OnnxModel, save_model
from horizon_nn.builder.model_builder import ModelBuilder
from horizon_nn.executor import ORTExecutor

ROOT = Path('/work/bpu/fusion_l128')
OUT = ROOT/'a16_w8_sm8'
OUT.mkdir(exist_ok=True)
report = {'status':'running','started_at':time.time(),'checks':[],
          'note':'Actual vendor quantized graph software validation, no board evidence'}
logging.basicConfig(level=logging.INFO)

def save():
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))

def main():
    source = json.loads((ROOT/'export_validation.json').read_text())
    assert source['status']=='complete', 'Export gate incomplete'
    graph = ROOT/'fusion_core_l128.onnx'
    identity = hashlib.sha256(graph.read_bytes()).hexdigest()
    provenance = OUT/'source_sha256.txt'
    if provenance.exists():
        assert provenance.read_text().strip()==identity, 'Existing checkpoint belongs to another export'
    else:
        provenance.write_text(identity)
    conf=yaml.safe_load(Path('/work/bpu/text_core_l32.yaml').read_text())
    conf['model_parameters'].update(onnx_model=str(graph),working_dir=str(OUT),output_model_file_prefix='embeddinggemma2_fusion_l128')
    conf['input_parameters']['input_shape']='1x512x1x128;1x1x1x128'
    conf['calibration_parameters']['cal_data_dir']=str(ROOT/'calibration/inputs_embeds')+';'+str(ROOT/'calibration/attention_mask')
    config=OUT/'config.yaml'
    config.write_text(yaml.safe_dump(conf,sort_keys=False))
    mapper=importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
    runner=mapper.MakertbinRunner(str(config),'onnx')
    runner._parse_conf()
    builder=ModelBuilder(OnnxModel(str(graph)),march='bayes-e',
        optimization=['set_Softmax_input_int8'],input_dict=runner._get_build_input_params(),
        cali_dict=runner._get_build_calibration_params())
    calibrated=OUT/'embeddinggemma2_fusion_l128_calibrated_model.onnx'
    quantized=OUT/'embeddinggemma2_fusion_l128_quantized_model.onnx'
    if calibrated.exists():
        model=OnnxModel(str(calibrated))
        report['calibration_reused']=True
    else:
        report['phase']='prepare_optimize_calibrate';save()
        original=builder.prepare(builder.input_model)
        save_model(original,str(OUT/'embeddinggemma2_fusion_l128_original_float_model.onnx'))
        optimized=builder.optimize(original)
        save_model(optimized,str(OUT/'embeddinggemma2_fusion_l128_optimized_float_model.onnx'))
        model,_,_=builder.calibrate(optimized)
        save_model(model,str(calibrated))
    protected=[]
    for node in model.graph.type2nodes['HzCalibration']:
        is_softmax='Softmax' in node.name or any('FROM_QUANTIZED_SOFTMAX' in consumer.name for consumer in node.next_ops)
        if node.tensor_type=='feature' and is_softmax:
            assert node.qtype=='int8',(node.name,node.qtype)
            protected.append(node.name)
        elif node.tensor_type=='feature':
            node.qtype='int16'
    assert protected
    report['protected_nodes']=protected
    if quantized.exists():
        quant=OnnxModel(str(quantized))
        report['quantization_reused']=True
    else:
        report['phase']='vendor_quantization';save()
        save_model(model,str(OUT/'embeddinggemma2_fusion_l128_mixed_calibrated_model.onnx'))
        quant=builder.quantize(model)
        save_model(quant,str(quantized))
    report['quantized_sha256']=hashlib.sha256(quantized.read_bytes()).hexdigest()
    report['phase']='software_validation';save()
    executor=ORTExecutor(quant).to('cpu').create_session()
    refs=np.load(ROOT/'reference_embeddings.npy')
    predictions=[]
    for row in source['checks']:
        index=row['index'];folder=ROOT/row['group']
        inputs={'inputs_embeds':np.fromfile(folder/f'inputs_embeds/{index:03d}.bin',np.float32).reshape(1,512,1,128),
                'attention_mask':np.fromfile(folder/f'attention_mask/{index:03d}.bin',np.float32).reshape(1,1,1,128)}
        raw=next(iter(executor.forward(inputs).values())).reshape(768)
        assert np.isfinite(raw).all() and np.linalg.norm(raw)>0
        pred=raw/np.linalg.norm(raw)
        cosine=float(pred@refs[index])
        result={'case':row['case'],'index':index,'group':row['group'],'cosine':cosine}
        if row['case'].endswith('_quantized_vision'):
            # Evaluate accumulated vision + fusion error against the untouched
            # full-model embedding, rather than checking the two stages alone.
            base=row['case'].removesuffix('_quantized_vision')
            semantic=Path('/work/bpu/results/multimodal_semantic_diagnosis')
            full_reference=np.load(semantic/f'{base}_embeddings.npz')['original']
            captions=np.load(semantic/'caption_embeddings.npy')
            scores=pred@captions.T
            image_index=int(base[:3])
            result['joint_embedding_cosine']=float(pred@full_reference)
            result['caption_scores']=scores.tolist()
            result['caption_rankings']=np.argsort(-scores).tolist()
            result['top1_correct']=int(np.argmax(scores)) in [image_index,image_index+4]
        report['checks'].append(result);predictions.append(pred)
        print(result,flush=True);save()
    predictions=np.asarray(predictions)
    np.save(OUT/'predictions.npy',predictions)
    for group in ['text_calibration','text_heldout']:
        indices=[row['index'] for row in source['checks'] if row['case'].startswith(group)]
        assert len(indices)==16
        p=predictions[indices]
        top1=np.argmax(p[8:]@p[:8].T,axis=1)
        report[group+'_top1']=int(np.sum(top1==np.arange(8)))
    report['minimum_cosine']=min(row['cosine'] for row in report['checks'])
    multimodal=[row for row in report['checks'] if 'joint_embedding_cosine' in row]
    assert len(multimodal)==4
    report['minimum_joint_multimodal_cosine']=min(row['joint_embedding_cosine'] for row in multimodal)
    report['software_gate_passed']=report['minimum_cosine']>=.95 and all(report[g+'_top1']==8 for g in ['text_calibration','text_heldout']) and all(
        row['joint_embedding_cosine']>=.95 and row['top1_correct'] for row in multimodal)
    report.update(status='complete',finished_at=time.time());save()

if __name__=='__main__':
    try:
        save();main()
    except Exception as error:
        report.update(status='failed',error=repr(error),finished_at=time.time());save()
        raise
