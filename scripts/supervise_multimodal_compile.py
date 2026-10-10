"""Finite continuation at software and placement gates; no automatic retry."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--resume-completed',action='store_true')
parser.add_argument('--resume-subgraphs',action='store_true')
args=parser.parse_args()
root=Path('/work/bpu')
logdir=root/'logs/multimodal_repair'
path=logdir/'compile_supervisor.json'
state={'status':'waiting_joint_gate','started_at':time.time(),'scopes':{}}

def save():
    state['updated_at']=time.time()
    path.write_text(json.dumps(state,indent=2))

try:
    save()
    if args.resume_completed:
        state['status']='verifying_vision_provenance';save()
        with (logdir/'vision_checkpoint_replay.log').open('w') as log:
            verified=subprocess.run(['python3.10','-u','/work/scripts/verify_vision_checkpoint.py'],
                stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        (logdir/'vision_checkpoint_replay.exit').write_text(str(verified.returncode))
        assert verified.returncode==0, 'Legacy vision provenance replay failed'
    deadline=time.time()+4200
    gate_exit=logdir/'joint_gate.exit'
    while not gate_exit.exists():
        if time.time()>deadline:
            raise TimeoutError('Joint gate wait deadline')
        time.sleep(15)
    assert gate_exit.read_text().strip()=='0', 'Joint validation process failed; inspect its log'
    gate=json.loads((root/'fusion_l128/a16_w8_sm8/validation.json').read_text())
    if not gate.get('software_gate_passed') or not gate.get('joint_multimodal_gate_passed'):
        state.update(status='software_accuracy_failed',report='bpu/fusion_l128/a16_w8_sm8/validation.json')
        save()
    else:
        for scope in ['fusion','vision']:
            folder=logdir/f'{scope}_compile'
            previous=folder/'state.json'
            if args.resume_completed and previous.exists():
                compiled=json.loads(previous.read_text())
                if compiled['status']=='complete':
                    assert not compiled['cpu_softmax_nodes']
                    runtime=Path(compiled['runtime'])
                    assert hashlib.sha256(runtime.read_bytes()).hexdigest()==compiled['runtime_sha256']
                    quantized=runtime.with_name(runtime.stem+'_quantized_model.onnx')
                    assert hashlib.sha256(quantized.read_bytes()).hexdigest()==compiled['quantized_sha256']
                    assert scope=='fusion' and compiled['quantized_sha256']==gate['quantized_sha256']
                    compiled['reused_after_hash_verification']=True
                    state['scopes'][scope]=compiled;save()
                    continue
            state['status']='compiling'
            state['scopes'][scope]={'status':'running'}
            save()
            folder.mkdir(exist_ok=True)
            with (folder/'compile.log').open('w') as log:
                command=['python3.10','-u','/work/scripts/compile_multimodal_candidate.py',scope]
                if args.resume_subgraphs and scope=='vision':
                    command.append('--resume-subgraphs')
                result=subprocess.run(command,
                    stdout=log,stderr=subprocess.STDOUT,timeout=14400 if args.resume_subgraphs else 5400)
            (folder/'compile.exit').write_text(str(result.returncode))
            assert result.returncode==0, f'{scope} compilation failed; no retry'
            compiled=json.loads((folder/'state.json').read_text())
            assert compiled['status']=='complete' and not compiled['cpu_softmax_nodes']
            state['scopes'][scope]=compiled
            save()
        state.update(status='compiled_candidates_ready',finished_at=time.time())
        save()
except Exception as error:
    state.update(status='failed',error=repr(error),finished_at=time.time());save()
    raise
