"""Compile one diagnostic slice only after the full 12-case semantic gate.

Verify six slice boundaries against the already tested two-layer candidate,
save independent HBM caches, and check Softmax/target Add BPU placement.
No board deployment, benchmark, weight change or automatic retry here.
"""
import gc
import argparse
import hashlib
import importlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import yaml
from horizon_nn.common import HbdkDictParser, infer_shapes, convert_reshape_target_shape_to_positive
from horizon_nn.executor import ORTExecutor
from horizon_nn.ir import OnnxModel, extract_submodel, save_model

ROOT = Path('/work/bpu/vision_trim576')
SOURCE_DIR = ROOT/'mlp_boundary_full16'
OUT = SOURCE_DIR/'chunk0_compile'
LOGS = ROOT.parent/'logs/multimodal_repair'
NAME = 'embeddinggemma2_vision_mlp_boundary_chunk0'
BOUNDARY = '/layers.1/Add_1_output_0_reshape_transpose'
parser = argparse.ArgumentParser()
parser.add_argument('--chunk',type=int,choices=range(8),default=0)
args = parser.parse_args()
if args.chunk:
    OUT = SOURCE_DIR/f'chunk{args.chunk}_compile'
    NAME = f'embeddinggemma2_vision_mlp_boundary_chunk{args.chunk}'
    BOUNDARY = f'/layers.{args.chunk*2+1}/Add_1_output_0_reshape_transpose' if args.chunk<7 else 'vision_features'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence = json.loads((SOURCE_DIR/'validation.json').read_text())
    assert (LOGS/'mlp_boundary_full16.exit').read_text().strip() == '0'
    assert evidence['status'] == 'complete' and evidence['joint_gate_passed']
    assert len(evidence['joint_checks']) == 12 and len(evidence['checks']) == 6
    assert all(c['cosine'] >= .95 and c['top1_correct'] for c in evidence['joint_checks'])
    assert evidence['quantized_conv_weights_byte_identical'] and evidence['softmax_named_nodes_byte_identical']
    source = SOURCE_DIR/'candidate_quantized.onnx'
    assert sha(source) == evidence['quantized_sha256']
    OUT.mkdir(exist_ok=False)
    state = dict(status='running', phase='partition', started_at=time.time(),
                 source_sha256=sha(source), board_tested=False, checks=[])
    destination = OUT/'state.json'

    def save():
        destination.write_text(json.dumps(state, indent=2))

    save()
    try:
        if args.chunk:
            partitions = json.loads((SOURCE_DIR/'split2_proto_verified/validation.json').read_text())
            assert partitions['status']=='complete' and partitions['software_equivalence_passed']
            assert partitions['source_sha256']==sha(source)
            assert len(partitions['checks'])==6 and all(c['max_abs']<1e-4 and c['relative_rmse']<1e-5 for c in partitions['checks'])
            spec = partitions['parts'][args.chunk]
            assert sha(spec['path'])==spec['sha256']
            path=OUT/f'{NAME}_quantized_model.onnx'
            shutil.copy2(spec['path'],path)
            state.update(partition_sha256=sha(path),checks=partitions['checks'],software_slice_equivalence_passed=True)
            return compile_runtime(path,state,save,spec['input'])
        model = infer_shapes(OnnxModel(str(source)), OnnxModel(str(
            ROOT/'a16_w8_sm8/embeddinggemma2_vision_trim576_original_float_model.onnx')), input_shape={})
        model = convert_reshape_target_shape_to_positive(model)
        part = extract_submodel(model, input_vars=['pixel_values'], output_vars=[BOUNDARY], infer_shapes=False)
        original = {n.name:n.proto.SerializeToString() for n in model.graph.nodes}
        assert all(n.proto.SerializeToString() == original[n.name] for n in part.graph.nodes)
        path = OUT/f'{NAME}_quantized_model.onnx'
        save_model(part, str(path))
        state['partition_sha256'] = sha(path)
        del model, part, original
        gc.collect()
        state['phase'] = 'slice_equivalence'
        save()
        session = ORTExecutor(str(path)).to('cpu').create_session()
        data = np.load(ROOT/'split2/full_board_inputs/data.npz')
        cases = ['calibration_0','calibration_1'] + [f'cat_frame_{i:02d}' for i in range(1,5)]
        for index, label in enumerate(cases):
            y = session.forward({'pixel_values':data[f'pixels_{index}']})[BOUNDARY]
            reference = np.load(ROOT/f'mlp_boundary_probe2_shapes/{label}_candidate_boundary.npy')
            assert y.shape == reference.shape and np.isfinite(y).all()
            error = float(np.max(np.abs(y-reference)))
            relative = float(np.linalg.norm(y-reference)/np.linalg.norm(reference))
            state['checks'].append(dict(case=label,max_abs=error,relative_rmse=relative))
            save()
            assert error < 1e-4 and relative < 1e-5, (label,error,relative)
            print('SLICE_EQUIVALENT',state['checks'][-1],flush=True)
        del session, data
        gc.collect()
        state['software_slice_equivalence_passed'] = True
        compile_runtime(path,state,save,'pixel_values')
    except Exception as error:
        state.update(status='failed',error=repr(error),finished_at=time.time())
        save()
        raise


def compile_runtime(path,state,save,input_name):
    try:
        conf = yaml.safe_load((ROOT/'a16_w8_sm8_fresh/config.yaml').read_text())
        conf['model_parameters'].update(onnx_model=str(path),working_dir=str(OUT),output_model_file_prefix=NAME)
        conf['input_parameters']['input_name'] = input_name
        config = OUT/'config.yaml'
        config.write_text(yaml.safe_dump(conf,sort_keys=False))
        mapper = importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
        hb = importlib.import_module('horizon_nn.compiler.hybrid_build')
        runner = mapper.MakertbinRunner(str(config),'onnx')
        runner._parse_conf()
        builder = hb.HybridBuilder(OnnxModel(str(path)),'bayes-e',
            HbdkDictParser(runner._get_build_hbdk_params()),output_path=str(OUT))
        builder.model_info(str(OUT/'node_placement.log'))
        indices = set()
        for index in range(builder.num_submodel()):
            indices.update(builder.get_subgraph_nodes(index))
        nodes = list(builder.quantized_model.graph.node)
        soft = [(i,n) for i,n in enumerate(nodes) if 'FROM_QUANTIZED_SOFTMAX' in n.name
                and n.op_type not in ('HzQuantize','HzDequantize')]
        target_names = {f'/layers.{i}/mlp/act_fn/{op}' for i in range(args.chunk*2,args.chunk*2+2) for op in ['Mul_2','Add']}
        targets = [(i,n) for i,n in enumerate(nodes) if n.name in target_names]
        state.update(phase='compile',subgraphs=builder.num_submodel(),softmax_nodes=len(soft),
            cpu_softmax_nodes=[n.name for i,n in soft if i not in indices],
            target_nodes=len(targets),cpu_target_nodes=[n.name for i,n in targets if i not in indices])
        save()
        assert len(soft)==14 and not state['cpu_softmax_nodes']
        assert len(targets)==4 and not state['cpu_target_nodes']
        checkpoint = OUT/'checkpoint'
        checkpoint.mkdir()
        compiler = Path('/usr/local/lib/python3.10/dist-packages/hbdk/bin/hbdk-cc')
        cache = dict(partition_sha256=sha(path),compiler_sha256=sha(compiler),entries={})
        original_compile = hb.hbdk_cc

        def capture(hbir_model,hbm_model,march,params=None,dump_all_models=False):
            original_compile(hbir_model,hbm_model,march,params,dump_all_models)
            key = Path(hbir_model).stem
            for artifact in [Path(hbir_model),Path(hbm_model)]:
                shutil.copy2(artifact,checkpoint/artifact.name)
                assert sha(artifact)==sha(checkpoint/artifact.name)
            cache['entries'][key] = dict(hbir_sha256=sha(hbir_model),hbm_sha256=sha(hbm_model),
                                        march=march,params=params or [])
            (checkpoint/'manifest.json').write_text(json.dumps(cache,indent=2))
            state['compiled_subgraphs'] = len(cache['entries'])
            save()

        hb.hbdk_cc = capture
        hybrid = builder.hybrid_model()
        onnx.save(hybrid,str(OUT/f'{NAME}_hybrid_model.onnx'))

        def use_hybrid(onnx_file=None,**kwargs):
            assert Path(onnx_file).resolve()==path.resolve()
            return hybrid

        mapper.build_onnx = use_hybrid
        runner.run('1.24.3')
        runtime = OUT/f'{NAME}.bin'
        assert runtime.is_file() and runtime.stat().st_size > 0
        state.update(status='complete',phase='runtime_packaged',runtime=str(runtime),
                     runtime_sha256=sha(runtime),runtime_bytes=runtime.stat().st_size,finished_at=time.time())
        save()
    except Exception as error:
        state.update(status='failed',error=repr(error),finished_at=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
