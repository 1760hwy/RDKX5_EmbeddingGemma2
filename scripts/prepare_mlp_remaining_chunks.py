"""Exact two-layer partitions of the semantically gated MLP repair candidate."""
import gc
import hashlib
import json
import time
import shutil
import onnx
from pathlib import Path
import numpy as np
from horizon_nn.common import infer_shapes, convert_reshape_target_shape_to_positive
from horizon_nn.executor import ORTExecutor
from horizon_nn.ir import OnnxModel, extract_submodel, save_model

ROOT=Path('/work/bpu/vision_trim576/mlp_boundary_full16')
OUT=ROOT/'split2_proto_verified'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gate=json.loads((ROOT/'validation.json').read_text())
    probe=json.loads((ROOT.parents[1]/'results/mlp_boundary_chunk0_board_probe/summary.json').read_text())
    assert gate['status']=='complete' and gate['joint_gate_passed']
    assert probe['status']=='complete' and probe['accuracy_passed'] and probe['core100']['count']==100
    assert probe['core100']['mean_core_ms'] < 882.515/2, 'First chunk did not meet speed screen'
    source=ROOT/'candidate_quantized.onnx';assert sha(source)==gate['quantized_sha256']
    OUT.mkdir(exist_ok=False)
    report=dict(status='running',phase='partition',source_sha256=sha(source),started_at=time.time(),parts=[],checks=[],board_tested=False)
    def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2))
    try:
        save()
        model=infer_shapes(OnnxModel(str(source)),OnnxModel(str(ROOT.parent/'a16_w8_sm8/embeddinggemma2_vision_trim576_original_float_model.onnx')),input_shape={})
        model=convert_reshape_target_shape_to_positive(model)
        original={n.name:n.proto.SerializeToString() for n in model.graph.nodes}
        selected=set();first='pixel_values'
        for chunk in range(8):
            last=f'/layers.{chunk*2+1}/Add_1_output_0_reshape_transpose' if chunk<7 else 'vision_features'
            part=extract_submodel(model,input_vars=[first],output_vars=[last],infer_shapes=False)
            names={n.name for n in part.graph.nodes}
            assert not selected & names and all(n.proto.SerializeToString()==original[n.name] for n in part.graph.nodes)
            selected.update(names)
            path=OUT/f'chunk{chunk}_quantized.onnx';save_model(part,str(path))
            report['parts'].append(dict(chunk=chunk,path=str(path),sha256=sha(path),input=first,output=last))
            first=last;save()
        assert selected==set(original)
        first_path=Path(report['parts'][0]['path'])
        compiled_path=ROOT/'chunk0_compile/embeddinggemma2_vision_mlp_boundary_chunk0_quantized_model.onnx'
        assert sha(compiled_path)==json.loads((ROOT/'chunk0_compile/state.json').read_text())['partition_sha256']
        fresh=onnx.load(str(first_path));tested=onnx.load(str(compiled_path))
        # SDK extraction can reorder independent branches and initializer/value_info lists.
        for field in ['node','initializer','value_info']:
            assert {v.name:v.SerializeToString() for v in getattr(fresh.graph,field)}=={v.name:v.SerializeToString() for v in getattr(tested.graph,field)}
            del getattr(fresh.graph,field)[:]
            del getattr(tested.graph,field)[:]
        assert fresh.SerializeToString()==tested.SerializeToString()
        shutil.copy2(compiled_path,first_path)
        report['parts'][0]['sha256']=sha(first_path)
        report['first_chunk_named_bytes_identical']=True
        save()
        del model,part,original;gc.collect()
        report['phase']='software_equivalence';save()
        data=np.load(ROOT.parent/'split2/full_board_inputs/data.npz')
        cases=['calibration_0','calibration_1']+[f'cat_frame_{i:02d}' for i in range(1,5)]
        refs=OUT/'software';refs.mkdir()
        for index,label in enumerate(cases):np.save(refs/f'{label}_boundary0.npy',data[f'pixels_{index}'])
        del data
        for spec in report['parts']:
            session=ORTExecutor(spec['path']).to('cpu').create_session()
            for label in cases:
                x=np.load(refs/f'{label}_boundary{spec["chunk"]}.npy')
                y=session.forward({spec['input']:x})[spec['output']]
                assert np.isfinite(y).all()
                np.save(refs/f'{label}_boundary{spec["chunk"]+1}.npy',y)
            del session;gc.collect()
            report['completed_software_chunks']=spec['chunk']+1;save()
            print('SOFTWARE_CHUNK',spec['chunk'],flush=True)
        for label in cases:
            y=np.load(refs/f'{label}_boundary8.npy');ref=np.load(ROOT/f'{label}_candidate_latent.npy')
            error=float(np.max(np.abs(y-ref)));relative=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
            report['checks'].append(dict(case=label,max_abs=error,relative_rmse=relative));save()
            assert error<1e-4 and relative<1e-5,(label,error,relative)
        report.update(status='complete',software_equivalence_passed=True,finished_at=time.time());save()
    except Exception as error:
        report.update(status='failed',error=repr(error),finished_at=time.time());save();raise


if __name__=='__main__':main()
