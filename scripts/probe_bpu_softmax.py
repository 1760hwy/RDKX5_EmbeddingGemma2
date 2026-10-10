"""Test the installed X5 toolchain's explicit INT8 Softmax option."""
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

root = Path('/work/bpu/softmax_probe')
root.mkdir(exist_ok=True)
shape = [1, 4, 32, 32]
node = helper.make_node('Softmax', ['scores'], ['probabilities'], name='attention_softmax', axis=3)
graph = helper.make_graph([node], 'softmax_probe',
    [helper.make_tensor_value_info('scores', TensorProto.FLOAT, shape)],
    [helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, shape)])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)], ir_version=7)
onnx.checker.check_model(model)
onnx.save(model, root / 'softmax.onnx')
cal = root / 'calibration'
cal.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
for i in range(8):
    rng.normal(0, 2, shape).astype(np.float32).tofile(cal / f'{i:03d}.bin')
(root / 'softmax.yaml').write_text('''model_parameters:
  onnx_model: /work/bpu/softmax_probe/softmax.onnx
  march: bayes-e
  output_model_file_prefix: softmax_int8
  working_dir: /work/bpu/softmax_probe/compiled
input_parameters:
  input_name: scores
  input_type_train: featuremap
  input_layout_train: NCHW
  input_shape: 1x4x32x32
  input_type_rt: featuremap
  input_layout_rt: NCHW
  norm_type: no_preprocess
calibration_parameters:
  cal_data_dir: /work/bpu/softmax_probe/calibration
  cal_data_type: float32
  calibration_type: max
  optimization: set_Softmax_input_int8
compiler_parameters:
  compile_mode: latency
  optimize_level: O3
  core_num: 1
  jobs: 4
''')
(root / 'metadata.json').write_text(json.dumps({'shape':shape,'axis':3,'seed':42},indent=2))
print('SOFTMAX_PROBE_READY', flush=True)
