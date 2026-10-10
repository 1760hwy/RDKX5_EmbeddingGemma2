"""Compile isolated diagnostic candidates after the joint semantic gate.

Vision's intermediate numerical failure is retained. This compilation tests
the final embedding hypothesis; it is not production acceptance.
"""
import argparse
import hashlib
import importlib
import json
import logging
from pathlib import Path
import onnx
import yaml
from horizon_nn.ir import OnnxModel
from horizon_nn.common import HbdkDictParser, infer_shapes, convert_reshape_target_shape_to_positive

parser=argparse.ArgumentParser()
parser.add_argument('scope',choices=['fusion','vision','vision_trim'])
parser.add_argument('--resume-subgraphs',action='store_true')
parser.add_argument('--verify-cache-only',action='store_true')
args=parser.parse_args()
scope=args.scope
root=Path('/work/bpu')
gate=root/'fusion_l128/a16_w8_sm8/validation.json'
report=json.loads(gate.read_text())
assert (root/'logs/multimodal_repair/joint_gate.exit').read_text().strip()=='0'
assert report['status']=='complete' and report['software_gate_passed'] and report['joint_multimodal_gate_passed']
if scope=='fusion':
    name='embeddinggemma2_fusion_l128'
    output=root/'fusion_l128/a16_w8_sm8'
    original=output
    config_source=output/'config.yaml'
elif scope=='vision':
    name='embeddinggemma2_vision_s70'
    output=root/'models/vision_a16_w8_sm8'
    original=root/'models/compiled_vision_s70_bounded'
    config_source=root/'vision_core_s70.yaml'
else:
    name='embeddinggemma2_vision_trim576'
    output=root/'vision_trim576/a16_w8_sm8_fresh'
    original=root/'vision_trim576/a16_w8_sm8'
    config_source=output/'config.yaml'
logdir=root/'logs/multimodal_repair'/f'{scope}_compile'
logdir.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
state={'status':'running','scope':scope,'acceptance':'diagnostic candidate only',
       'joint_software_gate':str(gate),'minimum_joint_cosine':report['minimum_joint_multimodal_cosine']}
if scope=='vision':
    intermediate=json.loads((root/'results/vision_mixed_precision_validation.json').read_text())
    state['intermediate_minimum_cosine']=intermediate['minimum_cosine']
    state['intermediate_numerical_gate_passed']=intermediate['numerical_gate_passed']
elif scope=='vision_trim':
    intermediate=json.loads((output/'validation.json').read_text())
    assert (root/'logs/multimodal_repair/vision_trim_fresh_quantize.exit').read_text().strip()=='0'
    assert intermediate['status']=='complete' and intermediate['joint_gate_passed']
    assert intermediate['fusion_quantized_sha256']==report['quantized_sha256']
    assert min(row['cosine'] for row in intermediate['joint_checks'])>=.95
    assert all(row['top1_correct'] for row in intermediate['joint_checks'])
    state['minimum_joint_cosine']=intermediate['minimum_joint_cosine']
    state['intermediate_minimum_cosine']=min(row['intermediate_cosine'] for row in intermediate['checks'])
    state['intermediate_numerical_gate_passed']=state['intermediate_minimum_cosine']>=.95
    state['vision_joint_software_gate']=str(output/'validation.json')

def save():
    (logdir/'state.json').write_text(json.dumps(state,indent=2))

try:
    save()
    # Never silently overwrite an earlier compiled result.
    assert not (output/f'{name}.bin').exists(), 'Runtime already exists; inspect its recorded hash before reuse'
    conf=yaml.safe_load(config_source.read_text())
    conf['model_parameters']['working_dir']=str(output)
    config=logdir/'config.yaml'
    config.write_text(yaml.safe_dump(conf,sort_keys=False))
    mapper=importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
    hb=importlib.import_module('horizon_nn.compiler.hybrid_build')
    runner=mapper.MakertbinRunner(str(config),'onnx')
    runner._parse_conf()
    quantized=output/f'{name}_quantized_model.onnx'
    state['quantized_sha256']=hashlib.sha256(quantized.read_bytes()).hexdigest()
    if scope=='fusion':
        assert state['quantized_sha256']==report['quantized_sha256']
    else:
        assert state['quantized_sha256']==intermediate['quantized_sha256']
    prepared=infer_shapes(OnnxModel(str(quantized)),OnnxModel(str(original/f'{name}_original_float_model.onnx')),input_shape={})
    prepared=convert_reshape_target_shape_to_positive(prepared)
    builder=hb.HybridBuilder(prepared,'bayes-e',HbdkDictParser(runner._get_build_hbdk_params()),output_path=str(output))
    builder.model_info(str(logdir/'node_placement.log'))
    state['subgraphs']=builder.num_submodel()
    nodes=list(builder.quantized_model.graph.node)
    bpu_indices=set()
    for i in range(builder.num_submodel()):
        bpu_indices.update(builder.get_subgraph_nodes(i))
    softmax=[(i,node) for i,node in enumerate(nodes) if 'FROM_QUANTIZED_SOFTMAX' in node.name and node.op_type not in ('HzQuantize','HzDequantize')]
    assert softmax, 'Missing quantized Softmax decomposition'
    state['softmax_nodes']=len(softmax)
    state['cpu_softmax_nodes']=[node.name for i,node in softmax if i not in bpu_indices]
    save()
    assert not state['cpu_softmax_nodes'], 'Softmax CPU fallback'
    if args.resume_subgraphs:
        assert scope=='vision'
        from compile_subgraph_cache import install, CachedPrefixVerified
        stats=install(hb,logdir/'checkpoint',state['quantized_sha256'],args.verify_cache_only)
        if args.verify_cache_only:
            try:
                builder.hybrid_model()
            except CachedPrefixVerified:
                pass
            assert len(stats['reused'])==len(json.loads((logdir/'checkpoint/resume_manifest.json').read_text())['entries'])
            (logdir/'checkpoint/verification.json').write_text(json.dumps({'status':'complete',**stats},indent=2))
            raise SystemExit(0)
    hybrid=builder.hybrid_model()
    onnx.save(hybrid,str(output/f'{name}_hybrid_model.onnx'))
    def use_hybrid(onnx_file=None,**kwargs):
        assert Path(onnx_file).resolve()==Path(runner.mp_conf.onnx_model).resolve()
        return hybrid
    mapper.build_onnx=use_hybrid
    runner.run('1.24.3')
    runtime=output/f'{name}.bin'
    assert runtime.is_file() and runtime.stat().st_size>0
    state.update(status='complete',runtime=str(runtime),runtime_bytes=runtime.stat().st_size,
                 runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
    save()
except Exception as error:
    state.update(status='failed',error=repr(error));save()
    raise
