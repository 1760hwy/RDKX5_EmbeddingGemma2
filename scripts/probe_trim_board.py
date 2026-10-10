"""Board loading gate: exact hashes, standalone vision, then co-residency.

This does not claim inference precision or speed. It never changes ION settings.
"""
import hashlib,json,subprocess,time
from pathlib import Path
from persistent_bpu import Runtime
root=Path(__file__).resolve().parents[1]
out=root/'bpu/results/vision_trim_load_probe'
out.mkdir(exist_ok=True)
report={'status':'running','started_at':time.time(),'inference_tested':False,'stages':[]}
vision=fusion=None
def save():(out/'summary.json').write_text(json.dumps(report,indent=2))
def inventory(label):
    # Installed sudo supports read-only debugfs inspection on this board.
    paths=[f'/sys/kernel/debug/ion/heaps/{name}' for name in ['carveout','cma_reserved','ion_cma']]
    result=subprocess.run(['sudo','-n','cat',*paths],capture_output=True,text=True,timeout=20)
    (out/f'{label}_ion.txt').write_text(result.stdout+result.stderr)
    return {'label':label,'inventory_exit':result.returncode}
try:
    save()
    spec=[('vision','bpu/vision_trim576/a16_w8_sm8_fresh/embeddinggemma2_vision_trim576.bin'),
          ('fusion','bpu/fusion_l128/a16_w8_sm8/embeddinggemma2_fusion_l128.bin')]
    compiled=json.loads((root/'bpu/runtime/vision_trim_deployment.json').read_text())
    hashes={m['relative_path']:m['sha256'] for m in compiled['models']}
    for scope,path in spec:
        actual=hashlib.sha256((root/path).read_bytes()).hexdigest()
        assert actual==hashes[path], 'Deployment hash mismatch'
    report['deployment']=compiled
    report['stages'].append(inventory('before'))
    report['phase']='standalone_vision_load';save();start=time.perf_counter()
    vision=Runtime(root/spec[0][1],[(1,768,1,576)])
    assert vision.output_shape==(1,512,1,64)
    report['stages'].append({'stage':'vision_loaded','load_seconds':time.perf_counter()-start})
    report['stages'].append(inventory('vision_loaded'));save()
    report['phase']='fusion_co_resident_load';save();start=time.perf_counter()
    fusion=Runtime(root/spec[1][1],[(1,512,1,128),(1,1,1,128)])
    assert fusion.output_shape==(1,768,1,1)
    report['stages'].append({'stage':'fusion_co_resident_loaded','load_seconds':time.perf_counter()-start})
    report['stages'].append(inventory('co_resident'));save()
    report.update(status='complete',standalone_vision_loaded=True,co_resident_loaded=True)
except Exception as error:
    report.update(status='failed',error=repr(error))
    report['stages'].append(inventory('failed'))
    raise
finally:
    if fusion:fusion.close()
    if vision:vision.close()
    report['stages'].append(inventory('released'))
    report['finished_at']=time.time();save()
