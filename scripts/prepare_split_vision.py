"""Extract two exact quantized halves; verify sequential software equivalence.

No recalibration, weight pruning or new quantization. This is a memory repair
candidate, not board acceptance. The F32 boundary is an existing CPU residual.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from horizon_nn.common import infer_shapes, convert_reshape_target_shape_to_positive
from horizon_nn.executor import ORTExecutor
from horizon_nn.ir import OnnxModel, extract_submodel, save_model

root = Path('/work/bpu/vision_trim576')
source = root/'a16_w8_sm8_fresh/embeddinggemma2_vision_trim576_quantized_model.onnx'
out = root/'split8'
out.mkdir(exist_ok=True)
boundary = '/layers.7/Add_1_output_0_reshape_transpose'
report = {'status': 'running', 'started_at': time.time(), 'board_tested': False,
          'boundary': boundary, 'checks': [], 'parts': []}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save():
    (out/'validation.json').write_text(json.dumps(report, indent=2))


try:
    save()
    assert digest(source) == 'bbaeb03aa9b6f756e5990fcfafe1582c619a693a994fbce5f0dae32b981b8a12'
    report['source_sha256'] = digest(source)
    model = infer_shapes(OnnxModel(str(source)), OnnxModel(str(
        root/'a16_w8_sm8/embeddinggemma2_vision_trim576_original_float_model.onnx')), input_shape={})
    model = convert_reshape_target_shape_to_positive(model)
    all_names = {n.name for n in model.graph.nodes}
    parts = [extract_submodel(model, input_vars=['pixel_values'], output_vars=[boundary], infer_shapes=False),
             extract_submodel(model, input_vars=[boundary], output_vars=['vision_features'], infer_shapes=False)]
    names = [{n.name for n in part.graph.nodes} for part in parts]
    assert not names[0] & names[1], 'Unexpected shared computational nodes'
    assert names[0] | names[1] == all_names, 'Partition omitted computational nodes'
    original_nodes = {n.name: n.proto.SerializeToString() for n in model.graph.nodes}
    for i, part in enumerate(parts):
        assert all(n.proto.SerializeToString() == original_nodes[n.name] for n in part.graph.nodes), 'Changed operator'
        path = out/f'part{i}_quantized.onnx'
        assert not path.exists(), 'Existing partition: inspect report before reusing'
        save_model(part, str(path))
        report['parts'].append({'index': i, 'path': str(path), 'sha256': digest(path),
                                'nodes': len(part.graph.nodes),
                                'inputs': [v.name for v in part.graph.inputs],
                                'outputs': [v.name for v in part.graph.outputs]})
    save()
    del model, parts, original_nodes
    gc.collect()
    inputs = []
    refs = np.load(root/'a16_w8_sm8_fresh/calibration_predictions.npy')
    for i in range(2):
        x = np.fromfile(root/f'calibration/vision_pixels/{i:03d}.bin', np.float32).reshape(1,768,1,576)
        inputs.append((f'calibration_{i}', x, refs[i].T[None,:,None,:]))
    heldout = root/'heldout_images_letterbox'
    for i in range(1,5):
        x = np.load(heldout/f'cat_frame_{i:02d}_image.npz')['pixels']
        ref = np.load(heldout/f'cat_frame_{i:02d}_vision_quantized.npy').T[None,:,None,:]
        inputs.append((f'cat_frame_{i:02d}', x, ref))
    first = ORTExecutor(str(out/'part0_quantized.onnx')).to('cpu').create_session()
    for label, x, _ in inputs:
        h = first.forward({'pixel_values': x})[boundary]
        assert h.shape == (1,768,1,576) and np.isfinite(h).all()
        np.save(out/f'{label}_boundary.npy', h)
        print('PART0_DONE', label, flush=True)
    del first
    gc.collect()
    second = ORTExecutor(str(out/'part1_quantized.onnx')).to('cpu').create_session()
    for label, _, ref in inputs:
        y = second.forward({boundary: np.load(out/f'{label}_boundary.npy')})['vision_features']
        assert y.shape == ref.shape and np.isfinite(y).all()
        error = float(np.max(np.abs(y-ref)))
        relative = float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
        cosine = float(y.ravel()@ref.ravel()/(np.linalg.norm(y)*np.linalg.norm(ref)))
        report['checks'].append({'case': label, 'max_abs_error': error,
                                 'relative_rmse': relative, 'cosine': cosine})
        np.save(out/f'{label}_prediction.npy', y)
        save()
        assert relative < 1e-5 and error < 1e-4, (label,error,relative)
        print('PART1_EQUIVALENT', report['checks'][-1], flush=True)
    report.update(status='complete', software_equivalence_passed=True, finished_at=time.time())
    save()
except Exception as error:
    report.update(status='failed', error=repr(error), finished_at=time.time())
    save()
    raise
