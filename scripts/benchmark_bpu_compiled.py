"""Run a successfully compiled candidate with the installed board runtime.

Float input layouts and dense output strides are verified before inference. Record all raw logs and
compare the exact calibration examples; these are not held-out accuracy tests.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('scope',choices=['text','vision'])
parser.add_argument('--candidate',choices=['baseline','a16_w8_sm8'],default='baseline')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
scope=args.scope
model_name={'text':'embeddinggemma2_text_l32','vision':'embeddinggemma2_vision_s70'}[scope]
model=root/'bpu/models'/f'{model_name}.bin'
out=root/'bpu/results'/f'{scope}_runtime'
if args.candidate!='baseline':
    model=root/'bpu/models'/f'{scope}_{args.candidate}'/f'{model_name}.bin'
    out=root/'bpu/results'/f'{scope}_{args.candidate}_runtime'
out.mkdir(parents=True,exist_ok=True)
def run(command,log,timeout=600):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    (out/log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{log}: exit {result.returncode}; see retained log')
    return result.stdout
base=['hrt_model_exec','--model_file',str(model)]
try:
    info=run([base[0],'model_info',*base[1:]],'model_info.log')
    sections=re.split(r'(?=input\[\d+\]:|output\[\d+\]:)',info)[1:]
    expected=([1,512,1,32],[1,1,1,32],[1,768,1,1]) if scope=='text' else ([1,768,1,630],[1,512,1,64])
    assert len(sections)==len(expected),(len(sections),len(expected))
    for index,(section,shape) in enumerate(zip(sections,expected)):
        found=re.search(r'valid shape:\s*\(([^)]+)\)',section)
        actual=[int(x) for x in found.group(1).split(',') if x.strip()]
        assert actual==shape,(actual,shape)
        assert 'HB_DNN_TENSOR_TYPE_F32' in section,section
        if index < len(expected)-1:
            assert 'HB_DNN_LAYOUT_NCHW' in section,section
        else:
            # This SDK tags the output NHWC while retaining ONNX dimension
            # order. Decode only verified dense strides, never from the tag.
            assert re.search(r'tensor layout: HB_DNN_LAYOUT_(NCHW|NHWC)',section),section
            strides=re.search(r'stride:\s*\(([^)]+)\)',section)
            actual_strides=[int(x) for x in strides.group(1).split(',') if x.strip()]
            dense=[4*int(np.prod(shape[i+1:])) for i in range(len(shape))]
            assert actual_strides==dense,(actual_strides,dense)
        aligned=re.search(r'aligned shape:\s*\(([^)]+)\)',section)
        assert [int(x) for x in aligned.group(1).split(',') if x.strip()]==shape,'Padding requires explicit decoding'
    refs=np.load(root/'bpu/results'/('reference_embeddings.npy' if scope=='text' else 'vision_reference_features.npy'))
    checks=[]
    predictions=[]
    for i,reference in enumerate(refs):
        files=([root/f'bpu/calibration/inputs_embeds/{i:03d}.bin',root/f'bpu/calibration/attention_mask/{i:03d}.bin']
            if scope=='text' else [root/f'bpu/calibration/vision_pixels/{i:03d}.bin'])
        dump=out/f'case_{i:03d}'
        dump.mkdir(exist_ok=True)
        log=run([base[0],'infer',*base[1:],'--input_file',','.join(map(str,files)),
            '--enable_dump','true','--dump_format','bin','--dump_path',str(dump)],f'case_{i:03d}.log')
        outputs=list(dump.glob('model_infer_output_0_*.bin'))
        assert len(outputs)==1,outputs
        raw=np.fromfile(outputs[0],dtype=np.float32)
        assert raw.size==reference.size,(raw.size,reference.size)
        assert np.isfinite(raw).all(),'Non-finite output'
        result=raw.reshape(768) if scope=='text' else raw.reshape(1,512,1,64).squeeze(2).transpose(0,2,1)[0]
        assert np.linalg.norm(result)>0,'Zero output'
        cosine=float(result.ravel()@reference.ravel()/(np.linalg.norm(result)*np.linalg.norm(reference)))
        checks.append({'case':i,'cosine':cosine,'max_abs_error':float(np.max(np.abs(result-reference))),
            'infer_ms':[float(x) for x in re.findall(r'Infer time:\s*([\d.]+)\s*ms',log)]})
        predictions.append(result)
        print('BOARD_OUTPUT',checks[-1],flush=True)
    np.save(out/'predictions.npy',np.asarray(predictions))
    heldout_report=None
    if scope=='text':
        evaluation=root/'bpu/evaluation'
        metadata=json.loads((evaluation/'metadata.json').read_text())
        assert metadata['status']=='complete'
        heldout_refs=np.load(evaluation/'reference_embeddings.npy')
        heldout_outputs=[]
        heldout_checks=[]
        for i,reference in enumerate(heldout_refs):
            dump=out/f'heldout_{i:03d}'
            dump.mkdir(exist_ok=True)
            files=[evaluation/f'inputs_embeds/{i:03d}.bin',evaluation/f'attention_mask/{i:03d}.bin']
            log=run([base[0],'infer',*base[1:],'--input_file',','.join(map(str,files)),
                '--enable_dump','true','--dump_format','bin','--dump_path',str(dump)],f'heldout_{i:03d}.log')
            outputs=list(dump.glob('model_infer_output_0_*.bin'))
            assert len(outputs)==1,outputs
            result=np.fromfile(outputs[0],dtype=np.float32)
            assert result.size==768 and np.isfinite(result).all() and np.linalg.norm(result)>0
            cosine=float(result@reference/(np.linalg.norm(result)*np.linalg.norm(reference)))
            heldout_checks.append({'case':i,'cosine':cosine,'max_abs_error':float(np.max(np.abs(result-reference)))})
            heldout_outputs.append(result/np.linalg.norm(result))
            print('HELDOUT_BOARD_OUTPUT',heldout_checks[-1],flush=True)
        heldout_outputs=np.asarray(heldout_outputs)
        ranking=np.argmax(heldout_outputs[8:]@heldout_outputs[:8].T,axis=1)
        heldout_report={'comparisons':heldout_checks,'minimum_cosine':min(x['cosine'] for x in heldout_checks),
            'top1':int(np.sum(ranking==np.arange(8))),'total':8,'indices':ranking.tolist(),
            'float_top1':metadata['float_retrieval_top1'],'note':metadata['note']}
        np.save(out/'heldout_predictions.npy',heldout_outputs)
    # Runtime perf includes CPU/BPU subgraphs and their scheduling. Input data
    # is generated by hrt_model_exec perf, not a retrieval accuracy sample.
    profile=out/'profile'
    profile.mkdir(exist_ok=True)
    perf=run([base[0],'perf',*base[1:],'--frame_count','100','--thread_num','1',
        '--profile_path',str(profile)],'perf.log',timeout=900)
    average=re.search(r'Average\s+latency\s+is:\s*([\d.]+)\s*ms',perf)
    assert average,'Missing runtime timing'
    report={'status':'complete','scope':scope,'model':str(model),'timestamp':time.time(),
        'comparisons':checks,'minimum_cosine':min(x['cosine'] for x in checks),
        'average_core_ms':float(average.group(1)),'perf_frames':100,'perf_threads':1,
        'quality_note':'Small calibration examples only; no independent production accuracy claim',
        'timing_note':'Whole compiled core including CPU/BPU transitions; excludes preprocessing and model loading'}
    if scope=='text':
        normalized=np.asarray(predictions)
        normalized/=np.linalg.norm(normalized,axis=1,keepdims=True)
        ranking=np.argmax(normalized[8:]@normalized[:8].T,axis=1)
        report.update(retrieval_top1=int(np.sum(ranking==np.arange(8))),retrieval_total=8,
            retrieval_indices=ranking.tolist(),heldout=heldout_report)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
except Exception as error:
    (out/'summary.json').write_text(json.dumps({'status':'failed','scope':scope,'error':str(error)},indent=2))
    raise
