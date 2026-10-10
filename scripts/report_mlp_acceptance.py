"""Audit the pulled final candidate measurements and render real-data figures."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/mlp_acceptance'
OUT.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))
cpu = read('bpu/results/mlp_batch_acceptance_cpu/summary.json')
bpu = read('bpu/results/mlp_batch_acceptance_bpu/summary.json')
assert cpu['status'] == bpu['status'] == 'complete'
assert cpu['manifest'] == bpu['manifest']
assert bpu['end_to_end_100_tested']
for mode, report in [('cpu', cpu), ('bpu', bpu)]:
    folder = ROOT / ('bpu/results/mlp_batch_acceptance_' + mode)
    checks = read(str(folder.relative_to(ROOT) / 'pull_verification.json'))
    for check in checks:
        assert check['matched']
        assert hashlib.sha256((folder / check['file']).read_bytes()).hexdigest() == check['sha256']
    assert report['all_sample_accuracy_passed'] and report['all_image_retrieval_passed']
assert len(list((ROOT / 'bpu/results/mlp_batch_acceptance_bpu').glob('*.npy'))) == 313
assert bpu['text']['count'] == 100
folder = ROOT / 'bpu/results/mlp_batch_acceptance_bpu'
caption_refs = np.load(str(ROOT / bpu['manifest']['caption_reference']))
data = np.load(str(ROOT / bpu['manifest']['gate_manifest']['data']))
captions = np.stack([np.load(str(folder / ('text_%03d_embedding.npy' % i))) for i in range(8)])
audited = 0
for row in bpu['samples']:
    name = 'text_cold_embedding.npy' if row['cold'] else 'text_%03d_embedding.npy' % row['repeat']
    pred = np.load(str(folder / name)).reshape(768)
    cosine = float(pred @ caption_refs[row['item']])
    assert np.isfinite(pred).all() and abs(np.linalg.norm(pred)-1) < 1e-5
    assert cosine >= .95 and abs(cosine-row['cosine']) < 2e-6
    audited += 1
for batch in bpu['batches']:
    for i, row in enumerate(batch['items']):
        pred = np.load(str(folder / ('%s_batch_%03d_item_%d_embedding.npy' % (batch['kind'], batch['repeat']+1, i)))).reshape(768)
        cosine = float(pred @ data['ref_%d_%s' % (row['item'], batch['kind'])])
        rank = np.argsort(-(captions @ pred)).tolist()
        assert cosine >= .95 and abs(cosine-row['cosine']) < 2e-6
        assert rank == row['ranking_board_captions']
        assert rank[0] in bpu['manifest']['gate_manifest']['images'][row['item']]['expected_caption_indices']
        assert np.isfinite(pred).all() and abs(np.linalg.norm(pred)-1) < 1e-5
        audited += 1
assert audited == 313
metrics = {'candidate': bpu['manifest']['gate_manifest'], 'modes': {}, 'stability': {}}
metrics['independently_audited_vectors'] = audited
for kind in ['text', 'image', 'mixed']:
    def wall(report):
        return report[kind]['mean_ms'] if kind == 'text' else report[kind]['weighted_ms_per_image']
    def core(report):
        if kind == 'text':
            rows = [row for row in report['samples'] if not row['cold']]
            return float(np.mean([sum(s['core_ms'] for s in row['stages']) for row in rows])) if all(row['stages'] for row in rows) else None
        batches = [row for row in report['batches'] if row['kind'] == kind and not row['cold']]
        return sum(sum(sum(s['core_ms_each']) for s in row['stages']) for row in batches) / sum(row['batch_size'] for row in batches)
    values = [row['cosine'] for row in bpu['samples']] if kind == 'text' else [item['cosine'] for batch in bpu['batches'] if batch['kind'] == kind for item in batch['items']]
    m = {'cpu_wall_ms': wall(cpu), 'bpu_wall_ms': wall(bpu), 'wall_speedup': wall(cpu)/wall(bpu),
         'cpu_core_ms': core(cpu), 'bpu_core_ms': core(bpu),
         'core_speedup': core(cpu)/core(bpu) if core(cpu) is not None else None, 'minimum_cosine': min(values)}
    if kind != 'text':
        warm = [row for row in bpu['batches'] if row['kind'] == kind and not row['cold']]
        assert sum(row['batch_size'] for row in warm) == 100
        six = [row['batch_wall_ms'] for row in warm if row['batch_size'] == 6]
        m.update(batch6_median_ms=float(np.median(six)), batch6_max_ms=max(six),
                 last_batch_size=warm[-1]['batch_size'], last_batch_wall_ms=warm[-1]['batch_wall_ms'])
    metrics['modes'][kind] = m
states = [bpu['state_before'], bpu['state_after']]
states += [r['state'] for r in bpu['samples'] if 'state' in r]
states += [r['state'] for r in bpu['batches']]
temps = [t for s in states for t in s['temperature_c'].values()]
rss = [s['rss_mb'] for s in states[2:]]
metrics['stability'] = {'temperature_min_c': min(temps), 'temperature_max_c': max(temps),
    'rss_sample_min_mb': min(rss), 'rss_sample_max_mb': max(rss),
    'observed_cpu_mhz': sorted(set(v for s in states for v in s['cpu_frequency_mhz']))}
metrics['initial_cpu_lookup_load_s'] = bpu['initial_model_load_seconds']
metrics['initial_fusion_load_s'] = bpu['initial_fusion_load_seconds']
metrics['text_cold_ms'] = bpu['samples'][0]['wall_ms']
single = {}
for mode in ['cpu','bpu']:
    single_folder = ROOT / ('bpu/results/mlp_single_probe_' + mode)
    if (single_folder / 'summary.json').exists():
        s = read(str(single_folder.relative_to(ROOT) / 'summary.json'))
        assert s['status'] == 'complete' and s['manifest'] == bpu['manifest']
        assert not s['end_to_end_100_tested'] and all(b['batch_size']==1 for b in s['batches'])
        checks = read(str(single_folder.relative_to(ROOT) / 'pull_verification.json'))
        assert all(c['matched'] and hashlib.sha256((single_folder/c['file']).read_bytes()).hexdigest()==c['sha256'] for c in checks)
        single[mode] = {k:s[k]['weighted_ms_per_image'] for k in ['image','mixed']}
metrics['single_request_small_probe'] = single
text_core_path = ROOT / 'bpu/results/mlp_text_core_probe_cpu/summary.json'
if text_core_path.exists():
    t = read(str(text_core_path.relative_to(ROOT)))
    tc_checks = read(str(text_core_path.parent.relative_to(ROOT) / 'pull_verification.json'))
    assert all(c['matched'] and hashlib.sha256((text_core_path.parent/c['file']).read_bytes()).hexdigest()==c['sha256'] for c in tc_checks)
    assert t['status']=='complete' and t['manifest']==bpu['manifest']
    rows = [r for r in t['samples'] if not r['cold']]
    assert len(rows)==8 and [r['item'] for r in rows]==list(range(8))
    cpu8 = float(np.mean([r['stages'][0]['core_ms'] for r in rows]))
    bpu8 = float(np.mean([r['stages'][0]['core_ms'] for r in bpu['samples'][1:9]]))
    metrics['cpu_text_core_diagnostic'] = {'cpu8_ms':cpu8,'bpu_matched8_ms':bpu8,'speedup':cpu8/bpu8,'scope':'eight matching captions, distinct CPU timing diagnostic, not CPU100'}
metrics['multimodal_core_screen_passed'] = all(metrics['modes'][k]['core_speedup'] >= 2 for k in ['image','mixed'])
metrics['precision_and_wall_screen_passed'] = all(m['wall_speedup'] >= 2 and m['minimum_cosine'] >= .95 for m in metrics['modes'].values())
metrics['all_core_screen_passed'] = metrics['multimodal_core_screen_passed'] and metrics.get('cpu_text_core_diagnostic',{}).get('speedup',0)>=2
(OUT / 'measured_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
labels = ['文本请求', '图片批处理吞吐', '图文批处理吞吐']
for ax, key, title in [(axes[0], 'wall', '完整流程：文本延迟 / 图像吞吐摊销'), (axes[1], 'core', '同配置计算核心耗时')]:
    kinds = ['text', 'image', 'mixed'] if key == 'wall' else ['image','mixed']
    x = np.arange(len(kinds))
    cv = [metrics['modes'][k]['cpu_' + key + '_ms']/1000 for k in kinds]
    bv = [metrics['modes'][k]['bpu_' + key + '_ms']/1000 for k in kinds]
    ax.bar(x-.18, cv, .36, color='#64748b', label='CPU FP32')
    ax.bar(x+.18, bv, .36, color='#0e7490', label='BPU mixed precision')
    ax.set_xticks(x); ax.set_xticklabels(labels if key=='wall' else labels[1:], fontproperties=FONT, fontsize=9)
    ax.set_title(title, fontproperties=FONT, fontsize=11)
    ax.set_ylabel('seconds'); ax.legend(fontsize=8, loc='center right')
    for i in range(len(kinds)):
        ax.text(i-.18, cv[i]+.15, '{:.2f}'.format(cv[i]), ha='center', fontsize=9)
        ax.text(i+.18, bv[i]+.15, '{:.2f}'.format(bv[i]), ha='center', fontsize=9)
fig.text(.5, .01, 'BPU每类100次；CPU文本8次、图像各18张。图像数值为六图调度吞吐，非单张延迟。', ha='center', fontproperties=FONT, fontsize=9)
fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(str(OUT/'speed_comparison.png'), dpi=160); plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot([r['wall_ms']/1000 for r in bpu['samples'] if not r['cold']], color='#0e7490')
axes[0].set_title('文本100次实际完整请求耗时', fontproperties=FONT)
axes[0].set_ylabel('seconds'); axes[0].set_xlabel('warm request')
for kind, color in [('image','#0e7490'), ('mixed','#b45309')]:
    batches = [r for r in bpu['batches'] if r['kind']==kind and not r['cold']]
    axes[1].plot([r['batch_wall_ms']/1000 for r in batches], label=kind, color=color, marker='o')
axes[1].set_title('每批真实等待时间（末批为四图）', fontproperties=FONT)
axes[1].set_ylabel('seconds per batch'); axes[1].legend()
fig.tight_layout(); fig.savefig(str(OUT/'raw_timing.png'), dpi=160); plt.close(fig)

canvas = Image.new('RGB', (1080, 600), 'white')
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
for i, entry in enumerate(bpu['manifest']['images']):
    img = Image.open(str(ROOT / entry['source'])).convert('RGB')
    img = ImageOps.contain(img, (340, 245))
    x,y = (i%3)*360, (i//3)*300
    canvas.paste(img, (x+(340-img.width)//2, y))
    draw.text((x+10,y+250), '样例{}：{}'.format(i+1, '同片猫帧' if i>1 else ['金门大桥','厨房绘画'][i]), font=font, fill='#172554')
canvas.save(str(OUT/'tested_images.png'))
print(json.dumps({k:v for k,v in metrics.items() if k!='candidate'}, ensure_ascii=False, indent=2))
