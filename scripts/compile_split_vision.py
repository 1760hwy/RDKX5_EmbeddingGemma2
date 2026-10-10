"""Compile exact quantized vision halves after the six-image equivalence gate.

Run without --part for finite supervision; never retry or overwrite a bin.
"""
import argparse
import gc
import hashlib
import importlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import onnx
import yaml
from horizon_nn.common import HbdkDictParser
from horizon_nn.ir import OnnxModel

root = Path('/work/bpu')
out = root/'vision_trim576/split8'
logs = root/'logs/multimodal_repair/vision_split_compile'
logs.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--part', type=int, choices=[0, 1])
parser.add_argument('--package-only', action='store_true')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def gate():
    assert (logs.parent/'vision_split_prepare.exit').read_text().strip() == '0'
    report = json.loads((out/'validation.json').read_text())
    assert report['status'] == 'complete' and report['software_equivalence_passed']
    assert len(report['checks']) == 6
    assert all(r['relative_rmse'] < 1e-5 and r['max_abs_error'] < 1e-4 for r in report['checks'])
    assert report['source_sha256'] == 'bbaeb03aa9b6f756e5990fcfafe1582c619a693a994fbce5f0dae32b981b8a12'
    for spec in report['parts']:
        assert sha(spec['path']) == spec['sha256']
    return report


def compile_part(index):
    evidence = gate()
    part = out/f'part{index}_quantized.onnx'
    target = out/f'part{index}'
    target.mkdir(exist_ok=True)
    name = f'embeddinggemma2_vision_split8_part{index}'
    runtime = target/f'{name}.bin'
    assert not runtime.exists(), 'Existing bin: inspect its state, do not overwrite'
    state = {'status': 'running', 'started_at': time.time(), 'part': index,
             'source_sha256': evidence['source_sha256'], 'partition_sha256': sha(part),
             'acceptance': 'memory repair candidate, not board acceptance'}
    destination = logs/f'part{index}_state.json'
    write(destination, state)
    try:
        conf = yaml.safe_load((root/'vision_trim576/a16_w8_sm8_fresh/config.yaml').read_text())
        conf['model_parameters'].update(onnx_model=str(part), working_dir=str(target),
                                        output_model_file_prefix=name)
        conf['input_parameters']['input_name'] = evidence['parts'][index]['inputs'][0]
        config = target/'config.yaml'
        config.write_text(yaml.safe_dump(conf, sort_keys=False))
        # MakertbinRunner.post_check loads this exact prefix relative to its
        # working directory even when build_onnx returns an existing hybrid.
        expected_quantized = target/f'{name}_quantized_model.onnx'
        if expected_quantized.exists():
            assert sha(expected_quantized) == sha(part)
        else:
            shutil.copy2(part, expected_quantized)
            assert sha(expected_quantized) == sha(part)
        mapper = importlib.import_module('horizon_tc_ui.hb_mapper_makertbin')
        hb = importlib.import_module('horizon_nn.compiler.hybrid_build')
        runner = mapper.MakertbinRunner(str(config), 'onnx')
        runner._parse_conf()
        model = OnnxModel(str(part))
        builder = hb.HybridBuilder(model, 'bayes-e', HbdkDictParser(runner._get_build_hbdk_params()),
                                   output_path=str(target))
        builder.model_info(str(logs/f'part{index}_node_placement.log'))
        bpu_indices = set()
        for i in range(builder.num_submodel()):
            bpu_indices.update(builder.get_subgraph_nodes(i))
        softmax = [(i, n) for i, n in enumerate(builder.quantized_model.graph.node)
                   if 'FROM_QUANTIZED_SOFTMAX' in n.name and n.op_type not in ('HzQuantize', 'HzDequantize')]
        state.update(subgraphs=builder.num_submodel(), softmax_nodes=len(softmax),
                     cpu_softmax_nodes=[n.name for i, n in softmax if i not in bpu_indices])
        write(destination, state)
        assert len(softmax) == 56 and not state['cpu_softmax_nodes'], 'Softmax placement changed'
        # Save newly generated per-subgraph artifacts immediately; original temp
        # files vanish on process exit. No cache is reused without verification.
        checkpoint = target/'checkpoint'
        checkpoint.mkdir(exist_ok=True)
        compiler = Path('/usr/local/lib/python3.10/dist-packages/hbdk/bin/hbdk-cc')
        manifest = {'partition_sha256': sha(part), 'compiler_sha256': sha(compiler), 'entries': {}}
        original_compile = hb.hbdk_cc

        def capture(hbir_model, hbm_model, march, params=None, dump_all_models=False):
            original_compile(hbir_model, hbm_model, march, params, dump_all_models)
            key = Path(hbir_model).stem
            for path in [Path(hbir_model), Path(hbm_model)]:
                shutil.copy2(path, checkpoint/path.name)
                assert sha(path) == sha(checkpoint/path.name)
            manifest['entries'][key] = {'hbir_sha256': sha(hbir_model), 'hbm_sha256': sha(hbm_model),
                                        'march': march, 'params': params or []}
            write(checkpoint/'manifest.json', manifest)

        hybrid_path = target/f'{name}_hybrid_model.onnx'
        if args.package_only:
            cached = json.loads((checkpoint/'manifest.json').read_text())
            assert cached['partition_sha256'] == sha(part) and cached['compiler_sha256'] == sha(compiler)
            assert len(cached['entries']) == builder.num_submodel()
            for key, entry in cached['entries'].items():
                assert sha(checkpoint/f'{key}.hbir') == entry['hbir_sha256']
                assert sha(checkpoint/f'{key}.hbm') == entry['hbm_sha256']
            hybrid = onnx.load(str(hybrid_path))
            names = {n.name for n in hybrid.graph.node if n.op_type == 'HzBpuHBM'}
            assert names == set(cached['entries']), 'Saved hybrid/cache graph mismatch'
            state.update(package_only=True, reused_hybrid_sha256=sha(hybrid_path))
            write(destination, state)
        else:
            hb.hbdk_cc = capture
            hybrid = builder.hybrid_model()
            onnx.save(hybrid, str(hybrid_path))

        def use_hybrid(onnx_file=None, **kwargs):
            assert Path(onnx_file).resolve() == part.resolve()
            return hybrid

        mapper.build_onnx = use_hybrid
        runner.run('1.24.3')
        assert runtime.is_file() and runtime.stat().st_size > 0
        state.update(status='complete', runtime=str(runtime), runtime_sha256=sha(runtime),
                     runtime_bytes=runtime.stat().st_size, finished_at=time.time())
        write(destination, state)
    except Exception as error:
        state.update(status='failed', error=repr(error), finished_at=time.time())
        write(destination, state)
        raise


def supervise():
    state = {'status': 'waiting_equivalence', 'started_at': time.time(), 'parts': []}
    path = logs/('resume_supervisor.json' if args.resume else 'supervisor.json')
    write(path, state)
    try:
        deadline = time.monotonic()+1800
        ready = logs.parent/'vision_split_prepare.exit'
        while not ready.exists():
            assert time.monotonic() < deadline, 'Equivalence wait exceeded 1800 seconds'
            time.sleep(15)
        gate()
        for index in [0, 1]:
            state.update(status='compiling', active_part=index)
            write(path, state)
            label = f'part{index}_package' if args.resume and index == 0 else f'part{index}'
            assert not (logs/f'{label}.log').exists(), 'Existing attempt log: inspect before running'
            command = [sys.executable, '-u', __file__, '--part', str(index)]
            if args.resume and index == 0:
                command.append('--package-only')
            with (logs/f'{label}.log').open('w') as stream:
                run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=300 if label.endswith('_package') else 7200)
            (logs/f'{label}.exit').write_text(str(run.returncode))
            assert run.returncode == 0, f'Part {index} compilation failed: {run.returncode}'
            result = json.loads((logs/f'part{index}_state.json').read_text())
            assert result['status'] == 'complete' and not result['cpu_softmax_nodes']
            state['parts'].append(result)
        assert sum(r['softmax_nodes'] for r in state['parts']) == 112
        state.update(status='complete', finished_at=time.time())
        write(path, state)
    except Exception as error:
        state.update(status='failed', error=repr(error), finished_at=time.time())
        write(path, state)
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    assert not args.package_only or args.part is not None
    if args.part is None:
        supervise()
    else:
        compile_part(args.part)
