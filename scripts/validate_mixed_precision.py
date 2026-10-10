"""Convert an isolated A16/W8/Softmax8 candidate and validate before compiling.

Uses installed vendor ModelBuilder.quantize; does not modify vendor files.
Software precision is a gate, never a claim about BPU placement or speed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from horizon_nn.ir import OnnxModel, save_model
from horizon_nn.builder.model_builder import ModelBuilder
from horizon_nn.executor import ORTExecutor

parser = argparse.ArgumentParser()
parser.add_argument('scope', choices=['text', 'vision'])
parser.add_argument('--reuse-quantized',action='store_true',help='Validate the existing isolated converted graph without repeating conversion')
args = parser.parse_args()
scope = args.scope
root = Path('/work/bpu')
prefix = 'embeddinggemma2_' + ('text_l32' if scope == 'text' else 'vision_s70')
original = root/'models'/('compiled_text_l32_bounded' if scope == 'text' else 'compiled_vision_s70_bounded')
candidate = root/'models'/f'{scope}_a16_w8_sm8'
candidate.mkdir(exist_ok=True)
destination = root/'results'/f'{scope}_mixed_precision_validation.json'
report = {'status':'running', 'scope':scope, 'precision':'A16/W8, protected Softmax INT8',
          'note':'Vendor quantized ONNX software gate; no compilation, BPU placement or speed claim',
          'started_at':time.time(), 'checks':[], 'protected_nodes':[]}
def persist():
    destination.write_text(json.dumps(report, indent=2))
persist()
try:
    model = OnnxModel(str(original/f'{prefix}_calibrated_model.onnx'))
    changed = []
    for node in model.graph.type2nodes['HzCalibration']:
        protected = 'Softmax' in node.name or any('FROM_QUANTIZED_SOFTMAX' in consumer.name for consumer in node.next_ops)
        if node.tensor_type == 'feature' and protected:
            assert node.qtype == 'int8', (node.name, node.qtype)
            report['protected_nodes'].append(node.name)
        elif node.tensor_type == 'feature':
            node.qtype = 'int16'
            changed.append(node.name)
    assert changed and report['protected_nodes']
    report['changed_nodes'] = changed
    report['phase'] = 'vendor_quantization'
    persist()
    quantized_path=candidate/f'{prefix}_quantized_model.onnx'
    if args.reuse_quantized:
        assert quantized_path.is_file(), 'No converted candidate to reuse'
        quantized=OnnxModel(str(quantized_path))
        report['conversion_reused']=True
    else:
        save_model(model, str(candidate/f'{prefix}_calibrated_model.onnx'))
        builder = ModelBuilder(model, march='bayes-e', optimization=['set_Softmax_input_int8'])
        quantized = builder.quantize(model)
        save_model(quantized, str(quantized_path))
    report['quantized_sha256']=hashlib.sha256(quantized_path.read_bytes()).hexdigest()
    report['quantized_model_saved'] = True
    report['phase'] = 'sample_validation'
    persist()
    executor = ORTExecutor(quantized).to('cpu').create_session()
    groups = [('calibration', root/'calibration', root/'results'/('reference_embeddings.npy' if scope=='text' else 'vision_reference_features.npy'))]
    if scope == 'text':
        groups.append(('heldout', root/'evaluation', root/'evaluation/reference_embeddings.npy'))
    for group, folder, ref_path in groups:
        refs = np.load(ref_path)
        predictions = []
        for i, ref in enumerate(refs):
            if scope == 'text':
                inputs = {'inputs_embeds':np.fromfile(folder/f'inputs_embeds/{i:03d}.bin',np.float32).reshape(1,512,1,32),
                          'attention_mask':np.fromfile(folder/f'attention_mask/{i:03d}.bin',np.float32).reshape(1,1,1,32)}
            else:
                inputs = {'pixel_values':np.fromfile(folder/f'vision_pixels/{i:03d}.bin',np.float32).reshape(1,768,1,630)}
            raw = next(iter(executor.forward(inputs).values()))
            assert raw.shape == ((1,768,1,1) if scope=='text' else (1,512,1,64)), raw.shape
            result = raw.reshape(768) if scope=='text' else raw.squeeze(2).transpose(0,2,1)[0]
            assert np.isfinite(result).all() and np.linalg.norm(result)>0
            cosine = float(result.ravel()@ref.ravel()/(np.linalg.norm(result)*np.linalg.norm(ref)))
            row={'group':group,'case':i,'cosine':cosine,'relative_rmse':float(np.linalg.norm(result-ref)/np.linalg.norm(ref))}
            predictions.append(result)
            report['checks'].append(row)
            persist()
            print(row,flush=True)
        predictions=np.asarray(predictions)
        np.save(candidate/f'{group}_predictions.npy',predictions)
        if scope == 'text':
            normalized=predictions/np.linalg.norm(predictions,axis=1,keepdims=True)
            top1=np.argmax(normalized[8:]@normalized[:8].T,axis=1)
            report[f'{group}_top1']=int(np.sum(top1==np.arange(8)))
            report[f'{group}_top1_indices']=top1.tolist()
    report['minimum_cosine']=min(row['cosine'] for row in report['checks'])
    report['numerical_gate_passed']=report['minimum_cosine']>=0.95 and (scope!='text' or all(report[f'{group}_top1']==8 for group,_,_ in groups))
    report.update(status='complete',finished_at=time.time())
    persist()
except Exception as error:
    report.update(status='failed',error=str(error),finished_at=time.time())
    persist()
    raise
