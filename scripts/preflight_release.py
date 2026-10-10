"""Read-only artifact-contract check. Does not import torch or start inference."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--bpu', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    errors = []
    checked = []

    def require(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            errors.append('Path outside root: ' + str(relative))
            return None
        if not path.is_file():
            errors.append('Missing: ' + str(relative))
            return None
        checked.append(str(relative))
        return path

    for name in ('model.safetensors', 'config.json', 'modules.json',
                 'tokenizer.json', 'config_sentence_transformers.json'):
        require('model/' + name)
    if args.bpu:
        manifest = require('bpu/vision_trim576/mlp_boundary_full16/full_board_inputs/pipeline_manifest.json')
        if manifest:
            try:
                gate = json.loads(manifest.read_text(encoding='utf-8'))['gate_manifest']
                specs = gate['vision'] + [gate['fusion']]
                if len(gate['vision']) != 8:
                    errors.append('Expected eight vision chunks')
                for spec in specs:
                    path = require(spec['path'])
                    if path:
                        digest = hashlib.sha256()
                        with path.open('rb') as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b''):
                                digest.update(block)
                        if digest.hexdigest() != spec['sha256']:
                            errors.append('SHA mismatch: ' + spec['path'])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append('Invalid manifest: ' + str(exc))
        for path in ('bpu/runtime/libembedding_bpu.so', 'model/processor_config.json',
                     'model/preprocessor_config.json', 'samples/GoldenGate.png',
                     'samples/kitchen_painting.jpg', 'samples/prepared/video_frames/frame_01.png'):
            require(path)
    print(json.dumps({'root': str(root), 'passed': not errors, 'checked': checked,
                      'errors': errors, 'inference_tested': False}, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
