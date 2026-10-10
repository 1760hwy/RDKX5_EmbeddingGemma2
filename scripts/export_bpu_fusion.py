"""Export a 128-token text/image fusion core from verified captured inputs.

No source weights are changed. Image encoding is separate; this core includes
PLE, all language layers, projection, masked mean pooling and normalization.
"""
import json
import os
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import onnx
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
BPU = ROOT/'bpu'
SOURCE = BPU/'results/multimodal_semantic_diagnosis'
OUT = BPU/'fusion_l128'
OUT.mkdir(exist_ok=True)
report = {'status': 'running', 'started_at': time.time(), 'sequence_length': 128,
          'note': 'FP32 export validation only; no quantization or board claim', 'checks': []}

def save():
    (OUT/'export_validation.json').write_text(json.dumps(report, indent=2))

class FusionCore(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds.squeeze(2).transpose(1, 2)
        valid = attention_mask.reshape(1, 128)
        additive = (1.0-valid[:, None, None, :]) * -32.0
        hidden = self.network(inputs_embeds=x, attention_mask={
            'full_attention': additive, 'sliding_attention': additive}).last_hidden_state
        mean = (hidden*valid.unsqueeze(-1)).sum(1)/valid.sum(1, keepdim=True).clamp(min=1)
        return torch.nn.functional.normalize(mean, p=2, dim=1).reshape(1, 768, 1, 1)

def main():
    save()
    semantic = json.loads((SOURCE/'summary.json').read_text())
    assert semantic['status'] == 'complete', 'Semantic diagnosis must finish first'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(ROOT/'model'), device='cpu', local_files_only=True,
        config_kwargs={'vision_config': None, 'audio_config': None},
        model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'}).eval()
    core = FusionCore(model[0].auto_model.language_model).eval()
    cases = []
    for path in sorted(SOURCE.glob('*_fusion_inputs.npz')):
        arrays = np.load(path)
        label = path.name.removesuffix('_fusion_inputs.npz')
        key = 'candidate' if '_quantized_vision' in label else 'reference'
        base = label.removesuffix('_quantized_vision')
        reference = np.load(SOURCE/f'{base}_embeddings.npz')[key]
        x = torch.from_numpy(arrays['inputs_embeds']).transpose(1, 2).unsqueeze(2)
        mask = torch.from_numpy(arrays['attention_mask']).float().reshape(1, 1, 1, -1)
        assert x.shape[-1] == mask.shape[-1] <= 128
        pad = 128-x.shape[-1]
        cases.append((label, torch.nn.functional.pad(x, (0, pad)),
                      torch.nn.functional.pad(mask, (0, pad)), reference))
    assert len(cases) == 8, len(cases)
    for group, folder, refs_path in [
        ('text_calibration', BPU/'calibration', BPU/'results/reference_embeddings.npy'),
        ('text_heldout', BPU/'evaluation', BPU/'evaluation/reference_embeddings.npy')]:
        refs = np.load(refs_path)
        for i, reference in enumerate(refs):
            x = torch.from_numpy(np.fromfile(folder/f'inputs_embeds/{i:03d}.bin', np.float32).reshape(1,512,1,32))
            mask = torch.from_numpy(np.fromfile(folder/f'attention_mask/{i:03d}.bin', np.float32).reshape(1,1,1,32))
            cases.append((f'{group}_{i:03d}', torch.nn.functional.pad(x,(0,96)),
                          torch.nn.functional.pad(mask,(0,96)), reference))
    for i, (label, x, mask, ref) in enumerate(cases):
        with torch.no_grad():
            prediction = core(x, mask).numpy().reshape(768)
        cosine = float(prediction @ ref)
        error = float(np.max(np.abs(prediction-ref)))
        assert cosine > .99999 and error < .0002, (label, cosine, error)
        # Held-out text remains separate and is never used for calibration.
        group = 'evaluation' if label.startswith('text_heldout') else 'calibration'
        for name, tensor in [('inputs_embeds',x), ('attention_mask',mask)]:
            folder = OUT/group/name
            folder.mkdir(parents=True, exist_ok=True)
            tensor.numpy().astype(np.float32).tofile(folder/f'{i:03d}.bin')
        report['checks'].append({'case':label,'index':i,'group':group,'tokens':int(mask.sum()),
                                  'wrapper_cosine':cosine,'wrapper_max_abs_error':error})
        save()
    path = OUT/'fusion_core_l128.onnx'
    torch.onnx.export(core, cases[0][1:3], str(path), input_names=['inputs_embeds','attention_mask'],
        output_names=['embedding'], opset_version=11, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(path)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    for row, (label, x, mask, ref) in zip(report['checks'], cases):
        pred = session.run(None, {'inputs_embeds':x.numpy(), 'attention_mask':mask.numpy()})[0].reshape(768)
        row['onnx_cosine'] = float(pred @ ref)
        row['onnx_max_abs_error'] = float(np.max(np.abs(pred-ref)))
        assert row['onnx_cosine'] > .99999 and row['onnx_max_abs_error'] < .0002, row
        print(row, flush=True)
        save()
    np.save(OUT/'reference_embeddings.npy', np.asarray([case[3] for case in cases]))
    report.update(status='complete', finished_at=time.time(), onnx_bytes=path.stat().st_size)
    save()

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        report.update(status='failed', error=repr(error), finished_at=time.time())
        save()
        raise
