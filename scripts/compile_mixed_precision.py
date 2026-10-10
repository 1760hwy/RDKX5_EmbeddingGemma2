"""Compile only a numerically accepted isolated candidate, using vendor APIs."""
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
parser.add_argument('scope',choices=['text','vision'])
args=parser.parse_args()
scope=args.scope
root=Path('/work/bpu')
report=json.loads((root/'results'/f'{scope}_mixed_precision_validation.json').read_text())
assert report['status']=='complete' and report['numerical_gate_passed'], 'Numerical gate failed; compilation prohibited'
name='embeddinggemma2_'+('text_l32' if scope=='text' else 'vision_s70')
output=root/'models'/f'{scope}_a16_w8_sm8'
original=root/'models'/('compiled_text_l32_bounded' if scope=='text' else 'compiled_vision_s70_bounded')
logdir=root/'logs'/f'{scope}_mixed_compile'
logdir.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
state={'status':'running','scope':scope,'precision_note':'Requested A16/W8/Softmax8; inspect placement and converter fallback warnings'}
def persist():
    (logdir/'state.json').write_text(json.dumps(state,indent=2))
persist()
try:
    conf=yaml.safe_load((root/('text_core_l32.yaml' if scope=='text' else 'vision_core_s70.yaml')).read_text())
    conf['model_parameters']['working_dir']=str(output)
    config=logdir/'config.yaml'
    config.write_text(yaml.safe_dump(conf,sort_keys=False))
    mapper=importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
    hb=importlib.import_module('horizon_nn.compiler.hybrid_build')
    runner=mapper.MakertbinRunner(str(config),'onnx')
    runner._parse_conf()
    quantized=output/f'{name}_quantized_model.onnx'
    prepared=infer_shapes(OnnxModel(str(quantized)),OnnxModel(str(original/f'{name}_original_float_model.onnx')),input_shape={})
    prepared=convert_reshape_target_shape_to_positive(prepared)
    builder=hb.HybridBuilder(prepared,'bayes-e',HbdkDictParser(runner._get_build_hbdk_params()),output_path=str(output))
    builder.model_info(str(logdir/'node_placement.log'))
    state['subgraphs']=builder.num_submodel()
    # Require every decomposed Softmax calculation to remain within BPU graphs.
    nodes=list(builder.quantized_model.graph.node)
    bpu_indices=set()
    for i in range(builder.num_submodel()):
        bpu_indices.update(builder.get_subgraph_nodes(i))
    softmax=[(i,n) for i,n in enumerate(nodes) if 'FROM_QUANTIZED_SOFTMAX' in n.name and n.op_type not in ('HzQuantize','HzDequantize')]
    assert softmax,'No quantized Softmax decomposition found'
    cpu_softmax=[n.name for i,n in softmax if i not in bpu_indices]
    state['softmax_nodes']=len(softmax)
    state['cpu_softmax_nodes']=cpu_softmax
    persist()
    assert not cpu_softmax,'Softmax CPU fallback violates candidate gate'
    hybrid=builder.hybrid_model()
    onnx.save(hybrid,str(output/f'{name}_hybrid_model.onnx'))
    def use_hybrid(onnx_file=None,**kwargs):
        assert Path(onnx_file).resolve()==Path(runner.mp_conf.onnx_model).resolve()
        return hybrid
    mapper.build_onnx=use_hybrid
    runner.run('1.24.3')
    runtime=output/f'{name}.bin'
    assert runtime.exists() and runtime.stat().st_size>0
    with runtime.open('rb') as stream:
        sha=hashlib.sha256()
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            sha.update(chunk)
    state.update(status='complete',runtime=str(runtime),runtime_bytes=runtime.stat().st_size,runtime_sha256=sha.hexdigest())
    persist()
except Exception as error:
    state.update(status='failed',error=str(error))
    persist()
    raise
