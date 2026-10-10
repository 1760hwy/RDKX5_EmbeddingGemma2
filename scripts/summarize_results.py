"""Generate a bounded CSV and measured-results report from finished runs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runs = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT / 'results').glob('*_t*.json'))]
completed = [r for r in runs if r.get('status') == 'complete']
fields = ['profile', 'threads', 'case', 'batch_size', 'actual_tokens', 'median_ms', 'p95_ms', 'items_per_second']
with (ROOT / 'results/benchmark_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in completed:
        for c in r['cases']:
            writer.writerow(dict(zip(fields, [r['profile'], r['threads'], c['name'], c['batch_size'],
                                  c.get('actual_tokens', []), round(c['median_ms'], 2),
                                  round(c['p95_ms'], 2), round(c['texts_per_second'], 3)])))
lines = ['# RDK X5 EmbeddingGemma 2 实测报告', '',
         '实验日期：2026-10-08。设备：192.0.2.10 / D-Robotics RDK X5 V1.0。', '',
         '使用官方 `google/embeddinggemma-2` 固定快照、CPU FP32；依赖、权重与脚本隔离在 `/home/sunrise/embeddinggemma2_lab`。', '',
         '设备为 8 核 Cortex-A55（最高 1.5 GHz），Ubuntu 22.04、约 6.9 GiB 内存。系统调频与后台服务保持原状。', '',
         f'已完成 {len(completed)} 组独立进程实验；结果来自真实板端推理。', '']
if len(completed) != len(runs):
    lines += ['注意：存在未完成 JSON，应检查对应日志；下表仅纳入完整实验。', '']
lines += ['## 加载与资源', '', '| 模式 | 线程 | 参数量 | 加载秒 | 加载后 RSS MB | 进程峰值 RSS MB | 最高采样温度 °C |',
          '|---|---:|---:|---:|---:|---:|---:|']
for r in completed:
    temps = [v for s in r.get('resource_samples', []) for v in s['temperature_c'].values()]
    lines.append(f"| {r['profile']} | {r['threads']} | {r['parameter_count']:,} | {r['load_seconds']:.2f} | {r['state_after_load']['rss_mb']:.0f} | {r['peak_rss_mb']:.0f} | {max(temps, default=0):.1f} |")
lines += ['', '加载计时从创建 SentenceTransformer 开始，不包含 Python 导入，未清空文件缓存。RSS 包含 Python、库、模型与临时张量；峰值是整个进程历史峰值。', '',
          '## 速度', '', '| 模式 | 线程 | 场景 | 批量 | 实际输入 token | median ms | 样本 p95 ms | 项/秒 |',
          '|---|---:|---|---:|---|---:|---:|---:|']
for r in completed:
    if r['profile'] == 'smoke':
        continue
    for c in r['cases']:
        tokens = ','.join(map(str, sorted(set(c.get('actual_tokens', []))))) or '见场景说明'
        lines.append(f"| {r['profile']} | {r['threads']} | {c['name']} | {c['batch_size']} | {tokens} | {c['median_ms']:.2f} | {c['p95_ms']:.2f} | {c['texts_per_second']:.3f} |")
best = ROOT / 'results/best_threads.json'
if best.exists():
    b = json.loads(best.read_text())
    lines += ['', f"本实验按四个单项文本场景延迟之和选出的线程数：**{b['threads']}**。不同批量、长度可能有不同最优值，应参考表格。"]
lines += ['', '计时包含分词、模型前向、池化、归一化与输出。文本预热两次，多模态预热一次；每组重复三次，原始样本见 JSON。媒体文件读取、音频解码及视频 FFmpeg 抽帧在计时前完成，模型内预处理计入。p95 基于少量重复样本，不能作为服务尾延迟 SLA。项/秒为批量大小除以该批次 median 秒数。', '',
          '## 功能检查', '', '所有完成实验均检查输出为 768 维、全部有限、L2 范数约为 1。']
for r in completed:
    if 'retrieval_smoke' in r:
        lines += ['', '| 输出维数 | 8 条查询 Top1 | MRR |', '|---:|---:|---:|']
        for dim, q in r['retrieval_smoke']['metrics'].items():
            lines.append(f"| {dim} | {q['top1_accuracy']:.1%} | {q['mrr']:.3f} |")
    if 'image_smoke' in r:
        lines += ['', f"合成红/蓝/绿方块的图文检索 Top1 索引：`{r['image_smoke']['top1']}`，目标索引为 `[0, 1, 2]`。"]
    for kind in ['image_retrieval', 'video_retrieval', 'audio_retrieval', 'text_image_retrieval']:
        if kind in r:
            q = r[kind]
            lines += ['', f'### {kind}', '', '| 输入编号 | 第一名候选 | 相似度 |', '|---:|---|---:|']
            for i, ranking in enumerate(q['rankings']):
                first = ranking[0]
                lines.append(f"| {i} | {q['captions'][first]} | {q['scores'][i][first]:.4f} |")
    if 'mixed_input' in r:
        lines += ['', f"文字+图片+音频混合输入：shape={r['mixed_input']['shape']}，finite={r['mixed_input']['finite']}，L2 norms={r['mixed_input']['norms']}。"]
lines += ['', '检索样例包含中文、英文以及中文查询检索英文文档。8 条人工编写查询及少量教程媒体属于功能检查，不代表 MTEB 或工业应用精度。相似度不是正确概率。缩短向量维数后重新 L2 归一化；缩短向量主要节省存储与相似度计算，不省去本实验模型前向。', '',
          '## 范围与限制', '',
          '- 本次测 CPU，没有 BPU 加速或板端量化结果。不能把 X5 的 BPU 算力换算成该模型速度。',
          '- 文本测试最长约 256 个内容 token；未验证 8192 token 上下文极限。',
          '- 多模态采用教程真实样例，视频仅取前 4 秒的 4 帧；未测试实时视频流、音视频同步和长媒体。',
          '- 未执行长时间稳定性、并发服务压力或真实业务语料精度评估。',
          '- 后台服务及 schedutil 调频会影响复测结果；频率、温度、可用内存快照保留在原始 JSON 中。', '',
          '## 证据与复现', '',
          '- `logs/environment.txt`：板端环境。',
          '- `logs/pip-freeze.txt`：实际安装版本。',
          '- `logs/model_manifest.json`：权重快照、大小和 SHA256；板端再次逐文件校验。',
          '- `logs/experiments.log` 与各组 `.log`：运行原始日志。',
          '- `results/*.json`：每次延迟样本、首次调用、资源采样与相似度矩阵。',
          '- `results/benchmark_summary.csv`：汇总表。',
          '- `scripts/run_experiments.sh`：完整复现入口；`scripts/infer.py`：离线查询/检索入口。', '',
          '官方：[模型卡](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)，[文本推理说明](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)，[官方权重](https://huggingface.co/google/embeddinggemma-2)。']
(ROOT / 'EXPERIMENT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote report for {len(completed)} completed runs')

media = ['# 多模态结果与样例', '', '本页只列已经完整运行的多模态实验。未完成项目见原始日志，不能当作通过。', '',
         '模型输出语义向量，用于检索、匹配；本实验不是生成图片描述或语音转写。', '',
         '![金门大桥](samples/GoldenGate.png)', '', '![厨房绘画](samples/kitchen_painting.jpg)', '',
         '[语音原文件](samples/roses-are.wav) · [猫咪视频](samples/cat_dozing.mp4)', '']
for r in completed:
    if not r['profile'].startswith('multimodal'):
        continue
    media += [f"## {r['mode']} / {r['threads']} 线程", '',
              f"加载 {r['load_seconds']:.2f} 秒；峰值进程 RSS {r['peak_rss_mb']:.0f} MB。", '',
              '| 输入 | 每批 median 秒 | 批量 | 首次调用秒 |', '|---|---:|---:|---:|']
    for c in r['cases']:
        media.append(f"| {c['name']} | {c['median_ms']/1000:.3f} | {c['batch_size']} | {c['first_call_s']:.3f} |")
    for kind in ['image_retrieval', 'video_retrieval', 'audio_retrieval', 'text_image_retrieval']:
        if kind not in r:
            continue
        q = r[kind]
        media += ['', f'### {kind}：前三名', '', '| 输入序号 | 名次 | 候选文本 | 余弦相似度 |', '|---:|---:|---|---:|']
        for i, ranking in enumerate(q['rankings']):
            for rank, idx in enumerate(ranking[:3], 1):
                media.append(f"| {i} | {rank} | {q['captions'][idx]} | {q['scores'][i][idx]:.4f} |")
    if 'mixed_input' in r:
        media += ['', f"文字+图片+音频：输出 {r['mixed_input']['shape']}；有限数值 {r['mixed_input']['finite']}；范数 {r['mixed_input']['norms']}。只验证混合输入能编码，不代表融合语义准确。"]
media += ['', '图片编号 0=金门大桥、1=厨房绘画。视频仅前 4 秒、4 帧；抽帧与音频解码不计入 encode 耗时。每项预热一次、重复三次，CPU FP32，无 BPU 加速。候选集合很小，结果不能推断通用多模态精度。']
(ROOT / 'MULTIMODAL_RESULTS.md').write_text('\n'.join(media) + '\n', encoding='utf-8')
