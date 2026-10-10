"""Repack two-layer slices from verified compiled halves, without hbdk-cc.

Keep all computational nodes and original compiled HBM bytes. Verify the six
software tensors before packaging; board loading/accuracy is a separate gate.
"""
import argparse
import gc
import hashlib
import importlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import yaml
from horizon_nn.executor import ORTExecutor
from horizon_nn.ir import OnnxModel, extract_submodel, save_model

parser = argparse.ArgumentParser()
parser.add_argument('--half', type=int, choices=[0,1], required=True)
parser.add_argument('--protobuf-hybrid', action='store_true')
args = parser.parse_args()
half = args.half
root = Path('/work/bpu/vision_trim576/split8')
source = root/f'part{half}'
name = f'embeddinggemma2_vision_split8_part{half}'
out = root.parent/'split2'/f"half{half}{'_proto' if args.protobuf_hybrid else ''}"
out.mkdir(parents=True, exist_ok=True)
report = {'status':'running','started_at':time.time(),'half':half,'phase':'extract',
          'board_tested':False,'checks':[],'parts':[]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    (out/'validation.json').write_text(json.dumps(report,indent=2))


def extract_hybrid(model, first, last):
    # hmct extract_submodel cannot copy the packed HBM STRING initializer
    # (it tries to create a random dtype=object array). Copy protobufs directly.
    graph=model.graph
    produced={v:n for n in graph.node for v in n.output}
    constants={t.name:t for t in graph.initializer}
    infos={v.name:v for v in [*graph.input,*graph.output,*graph.value_info]}
    pending=[last]
    required=set()
    nodes=set()
    while pending:
        value=pending.pop()
        if not value or value in required:
            continue
        required.add(value)
        if value==first or value in constants:
            continue
        assert value in produced, ('Unresolved activation',value)
        node=produced[value]
        nodes.add(node.name)
        pending.extend(node.input)
    selected=[n for n in graph.node if n.name in nodes]
    names=required|{v for n in selected for v in n.output}
    result=onnx.ModelProto()
    result.CopyFrom(model)
    for field in ['node','input','output','initializer','value_info']:
        result.graph.ClearField(field)
    result.graph.node.extend(selected)
    result.graph.initializer.extend(t for t in graph.initializer if t.name in required)
    result.graph.input.extend([infos[first]])
    result.graph.input.extend(v for v in graph.input if v.name in required and v.name in constants)
    result.graph.output.extend([infos[last]])
    inputs={v.name for v in result.graph.input}
    result.graph.value_info.extend(v for v in graph.value_info if v.name in names and v.name not in inputs and v.name!=last)
    onnx.checker.check_model(result)
    return result


try:
    save()
    evidence = json.loads((root/'validation.json').read_text())
    assert evidence['status']=='complete' and evidence['software_equivalence_passed']
    quantized = root/f'part{half}_quantized.onnx'
    assert sha(quantized)==evidence['parts'][half]['sha256']
    state = json.loads((Path('/work/bpu/logs/multimodal_repair/vision_split_compile')/f'part{half}_state.json').read_text())
    assert state['status']=='complete' and not state['cpu_softmax_nodes']
    assert sha(state['runtime'])==state['runtime_sha256']
    manifest = json.loads((source/'checkpoint/manifest.json').read_text())
    assert manifest['partition_sha256']==sha(quantized)
    hybrid_path = source/f'{name}_hybrid_model.onnx'
    report.update(source_quantized_sha256=sha(quantized),source_hybrid_sha256=sha(hybrid_path),
                  source_runtime_sha256=state['runtime_sha256'])
    hybrid_model = onnx.load(str(hybrid_path)) if args.protobuf_hybrid else OnnxModel(str(hybrid_path))
    quant_model = OnnxModel(str(quantized))
    cuts = [f'/layers.{half*8+i}/Add_1_output_0_reshape_transpose' for i in [1,3,5,7]]
    if half==1:
        cuts[-1]='vision_features'
    first = 'pixel_values' if half==0 else '/layers.7/Add_1_output_0_reshape_transpose'
    all_nodes = {n.name:n.SerializeToString() for n in hybrid_model.graph.node} if args.protobuf_hybrid else {n.name:n.proto.SerializeToString() for n in hybrid_model.graph.nodes}
    selected_nodes = set()
    for index, last in enumerate(cuts):
        folder = out/f'chunk{index}'
        folder.mkdir(exist_ok=True)
        chunk_name = f'embeddinggemma2_vision_split2_half{half}_chunk{index}'
        assert not (folder/f'{chunk_name}.bin').exists(), 'Existing runtime: inspect before reuse'
        qpart = extract_submodel(quant_model,input_vars=[first],output_vars=[last],infer_shapes=False)
        hpart = extract_hybrid(hybrid_model,first,last) if args.protobuf_hybrid else extract_submodel(hybrid_model,input_vars=[first],output_vars=[last],infer_shapes=False).proto
        names = {n.name for n in hpart.graph.node}
        assert not selected_nodes & names
        selected_nodes.update(names)
        assert all(n.SerializeToString()==all_nodes[n.name] for n in hpart.graph.node)
        qp = folder/f'{chunk_name}_quantized_model.onnx'
        save_model(qpart,str(qp))
        compiled_names = [n.name for n in hpart.graph.node if n.op_type=='HzBpuHBM']
        hbm_paths = []
        for key in compiled_names:
            entry = manifest['entries'][key]
            path = source/'checkpoint'/f'{key}.hbm'
            assert sha(path)==entry['hbm_sha256']
            assert sha(source/'checkpoint'/f'{key}.hbir')==entry['hbir_sha256']
            hbm_paths.append(str(path))
        assert compiled_names
        packed = folder/'packed_subset.hbm'
        subprocess.run(['/usr/local/lib/python3.10/dist-packages/hbdk/bin/hbdk-pack',
                        *hbm_paths,'-o',str(packed)],check=True,timeout=60)
        hp = hpart
        tensor = next(t for t in hp.graph.initializer if t.name=='PACKED_HBM_MODEL')
        tensor.CopyFrom(onnx.helper.make_tensor('PACKED_HBM_MODEL',onnx.TensorProto.STRING,[],[packed.read_bytes()]))
        hybrid_out = folder/f'{chunk_name}_hybrid_model.onnx'
        onnx.save(hp,str(hybrid_out))
        report['parts'].append({'chunk':index,'name':chunk_name,'input':first,'output':last,
                               'quantized':str(qp),'quantized_sha256':sha(qp),
                               'hybrid':str(hybrid_out),'hybrid_sha256':sha(hybrid_out),
                               'compiled_nodes':compiled_names,'packed_hbm_sha256':sha(packed)})
        first=last
        save()
    assert selected_nodes==set(all_nodes), 'Some compiled/CPU nodes omitted'
    del hybrid_model,quant_model,all_nodes,hpart,qpart
    gc.collect()
    # Each successive software boundary is saved without changing its dtype.
    cases=[f'calibration_{i}' for i in range(2)]+[f'cat_frame_{i:02d}' for i in range(1,5)]
    features={}
    for i,label in enumerate(cases):
        if half==1:
            features[label]=np.load(root/f'{label}_boundary.npy')
        elif i<2:
            features[label]=np.fromfile(root.parent/f'calibration/vision_pixels/{i:03d}.bin',np.float32).reshape(1,768,1,576)
        else:
            features[label]=np.load(root.parent/f'heldout_images_letterbox/{label}_image.npz')['pixels']
    report['phase']='software_equivalence'
    save()
    for spec in report['parts']:
        session=ORTExecutor(spec['quantized']).to('cpu').create_session()
        for label in cases:
            y=session.forward({spec['input']:features[label]})[spec['output']]
            assert y.dtype==np.float32 and np.isfinite(y).all()
            features[label]=y
        del session
        gc.collect()
        print('SOFTWARE_CHUNK_DONE',spec['chunk'],flush=True)
    for label in cases:
        ref=np.load(root/f'{label}_boundary.npy' if half==0 else root/f'{label}_prediction.npy')
        y=features[label]
        assert y.shape==ref.shape
        error=float(np.max(np.abs(y-ref)))
        relative=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
        assert error<1e-4 and relative<1e-5,(label,error,relative)
        report['checks'].append({'case':label,'max_abs_error':error,'relative_rmse':relative})
    report.update(software_equivalence_passed=True,phase='runtime_packaging')
    save()
    mapper=importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
    for spec in report['parts']:
        folder=Path(spec['quantized']).parent
        conf=yaml.safe_load((source/'config.yaml').read_text())
        conf['model_parameters'].update(onnx_model=spec['quantized'],working_dir=str(folder),
                                        output_model_file_prefix=spec['name'])
        conf['input_parameters']['input_name']=spec['input']
        config=folder/'config.yaml'
        config.write_text(yaml.safe_dump(conf,sort_keys=False))
        assert sha(spec['hybrid'])==spec['hybrid_sha256']
        saved_hybrid=onnx.load(spec['hybrid'])
        runner=mapper.MakertbinRunner(str(config),'onnx')
        runner._parse_conf()

        def use_hybrid(onnx_file=None,**kwargs):
            assert Path(onnx_file).resolve()==Path(spec['quantized']).resolve()
            return saved_hybrid

        mapper.build_onnx=use_hybrid
        runner.run('1.24.3')
        runtime=folder/f"{spec['name']}.bin"
        assert runtime.is_file()
        spec.update(runtime=str(runtime),runtime_sha256=sha(runtime),runtime_bytes=runtime.stat().st_size)
        save()
    report.update(status='complete',finished_at=time.time())
    save()
except Exception as error:
    report.update(status='failed',error=repr(error),finished_at=time.time())
    save()
    raise
