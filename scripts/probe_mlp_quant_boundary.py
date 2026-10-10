"""Software-only two-layer probe for profiled GELU polynomial CPU boundaries.

Reuse verified calibration. Add only two analytically bounded INT16 feature
calibrations after the constant coefficient Conv; no weight changes/compilation.
This diagnostic cannot establish final multimodal or board acceptance.
"""
import copy
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from horizon_nn.ir import OnnxModel, extract_submodel, save_model
from horizon_nn.builder.model_builder import ModelBuilder
from horizon_nn.executor import ORTExecutor

ROOT = Path('/work/bpu/vision_trim576')
args = argparse.ArgumentParser()
args.add_argument('--explicit-shapes', action='store_true')
args.add_argument('--full-model', action='store_true')
options = args.parse_args()
OUT = ROOT / ('mlp_boundary_full16' if options.full_model else
              'mlp_boundary_probe2_shapes' if options.explicit_shapes else 'mlp_boundary_probe2')
SOURCE = ROOT / 'a16_w8_sm8_fresh/embeddinggemma2_vision_trim576_mixed_calibrated_model.onnx'
BASE = ROOT / 'split2/half0_proto/chunk0/embeddinggemma2_vision_split2_half0_chunk0_quantized_model.onnx'
OUTPUT = '/layers.1/Add_1_output_0_reshape_transpose'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    report = dict(status='running', started_at=time.time(), source_sha256=sha(SOURCE),
                  baseline_sha256=sha(BASE), board_tested=False, compiled=False,
                  full_multimodal_accepted=False, edits=[], checks=[])

    def save():
        (OUT / 'validation.json').write_text(json.dumps(report, indent=2))

    save()
    try:
        if options.full_model:
            assert options.explicit_shapes
            prior = json.loads((ROOT / 'mlp_boundary_probe2_shapes/validation.json').read_text())
            assert prior['status'] == 'complete' and prior['diagnostic_gate_passed']
            assert sha(SOURCE) == prior['source_sha256']
            assert (ROOT.parent/'logs/multimodal_repair/mlp_boundary_probe2_shapes.exit').read_text().strip() == '0'
            m = onnx.load(SOURCE)
        else:
            m = extract_submodel(OnnxModel(str(SOURCE)), input_vars=['pixel_values'],
                                 output_vars=[OUTPUT], infer_shapes=False).proto
        layers = 16 if options.full_model else 2
        original = {n.name: n.SerializeToString() for n in m.graph.node}
        initializers = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
        weights_before = {t.name: t.SerializeToString() for t in m.graph.initializer}
        nodes = {n.name: n for n in m.graph.node}
        additions = {}
        changed = set()
        value_info = {v.name: v for v in m.graph.value_info}
        for layer in range(layers):
            prefix = f'/layers.{layer}/mlp/act_fn/'
            conv = nodes[prefix + 'Mul_2']
            add = nodes[prefix + 'Add']
            template = nodes[prefix + 'Mul_1_output_0_reshape_HzCalibration']
            weight_cal = nodes['UNIT_WEIGHT_FOR_' + prefix + 'Mul_2_HzCalibration']
            weight = initializers[weight_cal.input[0]]
            bias = initializers[conv.input[2]]
            assert conv.op_type == 'Conv' and np.all(bias == 0)
            assert weight.size and np.all(weight == weight.flat[0])
            attrs = {a.name: helper.get_attribute_value(a) for a in template.attribute}
            assert attrs['qtype'] == b'int16' and len(attrs['thresholds']) == 1
            threshold = float(attrs['thresholds'][0]) * abs(float(weight.flat[0]))
            assert threshold > 0 and np.isfinite(threshold)
            calibration = copy.deepcopy(template)
            calibration.name = prefix + 'Mul_2_profiled_boundary_HzCalibration'
            calibration.input[0] = conv.output[0]
            calibration.output[0] = conv.output[0] + '_profiled_int16_calibrated'
            if options.explicit_shapes:
                # Installed converter warned the new tensor had rank zero.
                # Calibration is shape preserving: copy the verified input type.
                info = copy.deepcopy(value_info[conv.output[0]])
                info.name = calibration.output[0]
                m.graph.value_info.append(info)
            for a in calibration.attribute:
                if a.name == 'thresholds':
                    a.CopyFrom(helper.make_attribute('thresholds', [threshold]))
                elif a.name == 'scales':
                    a.CopyFrom(helper.make_attribute('scales', [threshold / 32767]))
            assert add.input[1] == conv.output[0]
            add.input[1] = calibration.output[0]
            changed.add(add.name)
            additions[conv.name] = calibration
            report['edits'].append(dict(layer=layer, coefficient=float(weight.flat[0]),
                                        threshold=threshold, added_node=calibration.name))
        ordered = []
        for n in m.graph.node:
            ordered.append(copy.deepcopy(n))
            if n.name in additions:
                ordered.append(additions[n.name])
        del m.graph.node[:]
        m.graph.node.extend(ordered)
        assert all(n.SerializeToString() == original[n.name]
                   for n in m.graph.node if n.name in original and n.name not in changed)
        assert weights_before == {t.name: t.SerializeToString() for t in m.graph.initializer}
        onnx.checker.check_model(m)
        candidate = OUT / 'candidate_calibrated.onnx'
        onnx.save(m, str(candidate))
        report.update(phase='convert', weights_byte_identical=True,
                      changed_add_input_count=layers, explicit_calibration_shapes=options.explicit_shapes,
                      calibrated_sha256=sha(candidate))
        save()
        calibrated = OnnxModel(str(candidate))
        builder = ModelBuilder(calibrated, march='bayes-e', optimization=['set_Softmax_input_int8'])
        quant = builder.quantize(calibrated)
        qpath = OUT / 'candidate_quantized.onnx'
        save_model(quant, str(qpath))
        qnodes = {n.name: n for n in quant.proto.graph.node}
        report['add_mapping'] = {f'layer{i}': qnodes[f'/layers.{i}/mlp/act_fn/Add'].op_type
                                 for i in range(layers)}
        report['target_adds_quantized'] = all(v == 'HzSElementwiseAdd' for v in report['add_mapping'].values())
        report.update(phase='software_comparison', quantized_sha256=sha(qpath))
        save()
        if options.full_model:
            validate_full(qpath, report, save)
            report.update(status='complete', finished_at=time.time())
            save()
            return
        baseline = ORTExecutor(str(BASE)).to('cpu').create_session()
        candidate_session = ORTExecutor(str(qpath)).to('cpu').create_session()
        cases = [f'calibration_{i}' for i in range(2)] + [f'cat_frame_{i:02d}' for i in range(1, 5)]
        for i, case in enumerate(cases):
            if i < 2:
                x = np.fromfile(ROOT / f'calibration/vision_pixels/{i:03d}.bin', np.float32).reshape(1,768,1,576)
            else:
                x = np.load(ROOT / f'heldout_images_letterbox/{case}_image.npz')['pixels']
            y = candidate_session.forward({'pixel_values': x})[OUTPUT]
            ref = baseline.forward({'pixel_values': x})[OUTPUT]
            assert y.shape == ref.shape and np.isfinite(y).all()
            cosine = float(y.ravel() @ ref.ravel() / (np.linalg.norm(y) * np.linalg.norm(ref)))
            report['checks'].append(dict(case=case, cosine_to_original_quantized_boundary=cosine,
                                        max_abs=float(np.max(np.abs(y-ref))),
                                        relative_rmse=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))))
            np.save(OUT / f'{case}_candidate_boundary.npy', y)
            print(report['checks'][-1], flush=True)
            save()
        report['diagnostic_gate_passed'] = report['target_adds_quantized'] and all(
            c['cosine_to_original_quantized_boundary'] >= .999 for c in report['checks'])
        report.update(status='complete', finished_at=time.time(),
                      note='Two-layer provisional drift screen only; full FP32 semantic gate still required before compilation.')
        save()
    except Exception as error:
        report.update(status='failed', error=repr(error), finished_at=time.time())
        save()
        raise


def validate_full(qpath, report, save):
    """Original final FP32 references; unchanged fusion and real board captions."""
    baseline_path = ROOT / 'a16_w8_sm8_fresh/embeddinggemma2_vision_trim576_quantized_model.onnx'
    assert sha(baseline_path) == 'bbaeb03aa9b6f756e5990fcfafe1582c619a693a994fbce5f0dae32b981b8a12'
    old = onnx.load(baseline_path)
    new = onnx.load(qpath)
    by_name = {n.name: n for n in new.graph.node}
    soft = [n for n in old.graph.node if 'Softmax' in n.name or 'FROM_QUANTIZED_SOFTMAX' in n.name]
    report['softmax_named_nodes_byte_identical'] = bool(soft) and all(
        n.name in by_name and n.SerializeToString() == by_name[n.name].SerializeToString() for n in soft)
    report['softmax_named_nodes'] = len(soft)
    # All named quantized Conv weight tensors must remain byte-identical.
    old_init = {t.name: t.SerializeToString() for t in old.graph.initializer}
    new_init = {t.name: t.SerializeToString() for t in new.graph.initializer}
    weights = {n.input[1] for n in old.graph.node if n.op_type == 'HzSQuantizedConv'}
    report['quantized_conv_weights_byte_identical'] = bool(weights) and all(
        name in new_init and old_init[name] == new_init[name] for name in weights)
    save()
    del old, new, by_name, old_init, new_init
    manifest = json.loads((ROOT/'split2/full_board_inputs/manifest.json').read_text())
    gate_data = ROOT/'split2/full_board_inputs/data.npz'
    assert sha(gate_data) == manifest['data_sha256']
    data = np.load(gate_data)
    report['reference_data_sha256'] = sha(gate_data)
    vision = ORTExecutor(str(qpath)).to('cpu').create_session()
    features = []
    for index, case in enumerate(manifest['images']):
        pred = vision.forward({'pixel_values': data[f'pixels_{index}']})['vision_features']
        ref = data[f'latent_{index}']
        assert pred.shape == (1,512,1,64) and np.isfinite(pred).all()
        report['checks'].append(dict(case=case['case'],
            cosine_to_original_quantized_latent=float(pred.ravel() @ ref.ravel()/(np.linalg.norm(pred)*np.linalg.norm(ref))),
            relative_rmse=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref))))
        np.save(OUT/f'{case["case"]}_candidate_latent.npy', pred)
        features.append(pred.squeeze(2).transpose(0,2,1)[0])
        print(report['checks'][-1], flush=True)
        save()
    del vision
    import gc
    gc.collect()
    fusion_path = ROOT.parent/'fusion_l128/a16_w8_sm8/embeddinggemma2_fusion_l128_quantized_model.onnx'
    assert sha(fusion_path) == '34f48090551d69fd7b42050a91412bc7201a634fa0492d8afe5ee91880fe9a3e'
    fusion = ORTExecutor(str(fusion_path)).to('cpu').create_session()
    captions_path = ROOT.parent/'results/vision_split2_full_board_gate/board_caption_embeddings.npy'
    board_gate = json.loads((captions_path.parent/'summary.json').read_text())
    assert board_gate['status'] == 'complete' and board_gate['joint_board_gate_passed']
    assert board_gate['manifest']['fusion'] == manifest['fusion']
    captions = np.load(captions_path)
    report['caption_reference_source'] = 'Actual board captions from unchanged, SHA-locked fusion; not generated by new vision.'
    report['caption_reference_sha256'] = sha(captions_path)
    report['fusion_quantized_sha256'] = sha(fusion_path)
    report['joint_checks'] = []
    semantic = ROOT.parent/'results/multimodal_semantic_diagnosis'
    for index, case in enumerate(manifest['images']):
        for kind in ['image', 'mixed']:
            if index < 2:
                inp = np.load(semantic/f'{index:03d}_{kind}_fusion_inputs.npz')
                x = inp['inputs_embeds'].copy()
                original = np.load(ROOT.parent/'results/vision_reference_features.npy')[index]
                matches = np.max(np.abs(x[0,:,None,:]-original[None,:,:]),axis=2) < 1e-5
                assert np.all(matches.sum(axis=0) == 1)
                indices = np.argmax(matches,axis=0)
                assert len(np.unique(indices)) == 64
                x[0,indices] = features[index]
            else:
                inp = np.load(ROOT/f'heldout_images_letterbox/{case["case"]}_{kind}.npz')
                x = inp['inputs_embeds'].copy()
                x[inp['image_mask']] = features[index]
            length = x.shape[1]
            assert length <= 128
            padded = np.pad(x,((0,0),(0,128-length),(0,0))).transpose(0,2,1)[:,:,None,:]
            mask = np.pad(inp['attention_mask'],((0,0),(0,128-length)))[:,None,None,:]
            y = fusion.forward({'inputs_embeds':padded.astype(np.float32),
                                'attention_mask':mask.astype(np.float32)})
            value = next(iter(y.values())).reshape(768)
            assert np.isfinite(value).all() and np.linalg.norm(value) > 0
            value /= np.linalg.norm(value)
            ranking = np.argsort(-(value @ captions.T))
            row = dict(case=case['case']+'_'+kind, cosine=float(value @ data[f'ref_{index}_{kind}']),
                       caption_rankings=ranking.tolist(),
                       top1_correct=int(ranking[0]) in case['expected_caption_indices'])
            report['joint_checks'].append(row)
            np.save(OUT/f'{row["case"]}_candidate_embedding.npy',value)
            print(row,flush=True)
            save()
    report['minimum_joint_cosine'] = min(c['cosine'] for c in report['joint_checks'])
    report['joint_gate_passed'] = (len(report['joint_checks']) == 12 and report['target_adds_quantized']
        and report['softmax_named_nodes_byte_identical'] and report['quantized_conv_weights_byte_identical']
        and all(c['cosine'] >= .95 and c['top1_correct'] for c in report['joint_checks']))
    report['note'] = 'Software complete vision+unchanged fusion gate, no new compilation or board performance evidence.'


if __name__ == '__main__':
    main()
