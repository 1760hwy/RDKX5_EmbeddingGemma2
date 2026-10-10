"""Publish completed evidence and explicitly label unfinished experiments."""
import json
from datetime import datetime
from pathlib import Path

root=Path(__file__).resolve().parents[1]
results=root/'bpu/results'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
lines=['# RDK X5 EmbeddingGemma 2 测试结果', '', f'生成时间：{datetime.now().astimezone().isoformat()}', '',
       '首版编译核心有加速但精度失真。新候选的实际状态与板端结果见下文；不能把软件验证当作板端修复。', '',
       '## 首版真实板端速度与精度', '', '| 核心 | CPU FP32 中位数 | 板端100帧平均 | 核心加速 | 精度 |', '|---|---:|---:|---:|---|']
summary={'generated_at':datetime.now().astimezone().isoformat(),'baseline':{},'new_candidates':{}}
for scope in ['text','vision']:
    cpu=read(results/('cpu_text_core_l32.json' if scope=='text' else 'cpu_vision_core_s70.json'))
    board=read(results/f'{scope}_runtime/summary.json')
    assert cpu['status']=='complete' and board['status']=='complete'
    quality=(f"独立检索 {board['heldout']['top1']}/8；最低余弦 {board['heldout']['minimum_cosine']:.6f}" if scope=='text'
             else '两图特征余弦 '+', '.join(f"{row['cosine']:.6f}" for row in board['comparisons']))
    ratio=cpu['median_ms']/board['average_core_ms']
    lines.append(f"| {scope} | {cpu['median_ms']:.3f} ms | {board['average_core_ms']:.3f} ms | {ratio:.2f}× | {quality} |")
    summary['baseline'][scope]={'cpu_median_ms':cpu['median_ms'],'board_mean_ms':board['average_core_ms'],'speedup':ratio,'accuracy':quality}
lines += ['', 'CPU 与板端统计方法不同；均排除模型加载及核心外预处理。板端包括核心内部 CPU/BPU 调度。视觉核心不包含最终多模态文本网络，因此不是完整图文检索延迟。', '',
          '## 新混合精度候选', '', '设定为权重 INT8、其他激活 INT16、Softmax INT8。转换日志存在部分 INT16 输入回退，实际执行精度与分图需另行检查。', '',
          '| 核心 | 软件验证状态 | 已测样例 | 最低余弦 | 校准/独立检索 | 数值门槛 | 板端状态 |', '|---|---|---:|---:|---|---|---|']
for scope in ['text','vision']:
    report=read(results/f'{scope}_mixed_precision_validation.json')
    checks=report.get('checks',[])
    minimum=f"{min(row['cosine'] for row in checks):.6f}" if checks else '未产生'
    retrieval=f"{report.get('calibration_top1','待定')}/8，{report.get('heldout_top1','待定')}/8" if scope=='text' else '不适用'
    gate=report.get('numerical_gate_passed','待定')
    boardpath=results/f'{scope}_a16_w8_sm8_runtime/summary.json'
    boardstatus=read(boardpath)['status'] if boardpath.exists() else '未取得板端结果'
    lines.append(f"| {scope} | {report['status']} | {len(checks)} | {minimum} | {retrieval} | {gate} | {boardstatus} |")
    summary['new_candidates'][scope]={'status':report['status'],'samples':len(checks),'minimum_cosine':minimum,'numerical_gate':gate,'board_status':boardstatus}
lines += ['', '筛选门槛为所有固定样例余弦 ≥0.95、文本两组检索各8/8、核心加速至少2倍。这是小样本工程筛选，不是生产数据集验收。']
candidate_board=results/'text_a16_w8_sm8_runtime/summary.json'
if candidate_board.exists():
    board=read(candidate_board)
    if board['status']=='complete':
        cpu=read(results/'cpu_text_core_l32.json')
        compile_state=read(root/'bpu/logs/text_mixed_compile/state.json')
        ratio=cpu['median_ms']/board['average_core_ms']
        quality_pass=min(board['minimum_cosine'],board['heldout']['minimum_cosine'])>=0.95 and board['retrieval_top1']==8 and board['heldout']['top1']==8
        speed_pass=ratio>=2
        summary['new_candidates']['text'].update(board_mean_ms=board['average_core_ms'],speedup=ratio,
            board_minimum_cosine=min(board['minimum_cosine'],board['heldout']['minimum_cosine']),
            board_calibration_top1=board['retrieval_top1'],board_heldout_top1=board['heldout']['top1'],
            accuracy_gate_passed=quality_pass,speed_gate_passed=speed_pass,overall_gate_passed=quality_pass and speed_pass)
        lines += ['', '## 新文本候选真实板端结果（已完成并拉取）', '',
                  '| 指标 | 结果 |', '|---|---:|',
                  f"| 单线程100帧平均核心耗时 | {board['average_core_ms']:.3f} ms |",
                  f"| 核心吞吐 | {1000/board['average_core_ms']:.3f} 次/秒 |",
                  f"| 相对同形状CPU FP32核心加速 | {ratio:.2f}× |",
                  f"| 校准16样例最低余弦 | {board['minimum_cosine']:.6f} |",
                  f"| 独立16样例最低余弦 | {board['heldout']['minimum_cosine']:.6f} |",
                  f"| 校准/独立检索 | {board['retrieval_top1']}/8，{board['heldout']['top1']}/8 |",
                  f"| 精度门槛/速度门槛 | {quality_pass} / {speed_pass} |", '',
                  f"编译分图 {compile_state['subgraphs']} 个；{compile_state['softmax_nodes']} 个 Softmax 分解计算节点全部在 BPU，CPU Softmax 列表为空。原始日志及量化图含部分输入回退，其他CPU算子仍存在；整网不是纯INT8、纯BPU。", '',
                  f"运行模型大小 {compile_state['runtime_bytes']} 字节，SHA256 `{compile_state['runtime_sha256']}`。软件与板端校准输出逐元素一致；独立输出归一化后的最大差2.98e-8。", '',
                  '评估：文本精度在当前小样本上恢复，可用于低频短文本检索的试验原型；当前固定32 tokens，长文本分块、完整端到端耗时和真实业务集尚未验收。速度仅1.86倍，未达到本项目2倍筛选目标，不能宣称整体部署验收成功。视觉候选最低余弦0.927393，未达精度门槛，因此完整多模态应用仍不建议采用。', '',
                  '混合精度分图从首版25个增加至73个，是潜在调度开销方向，但尚未用逐算子耗时证明其为唯一瓶颈。未进行剪枝，也没有以剪枝替代精度验证。', '',
                  '原始新候选数据：bpu/results/text_a16_w8_sm8_runtime/summary.json、predictions.npy、heldout_predictions.npy、perf.log、各样例dump及profile；转换与分图记录 bpu/logs/text_mixed_compile/。']
lines += ['', '## 独立文本检索错误样例（首版板端）', '']
meta=read(root/'bpu/evaluation/metadata.json')
board=read(results/'text_runtime/summary.json')
for i,predicted in enumerate(board['heldout']['indices']):
    if predicted!=i:
        lines += [f"- 查询：{meta['cases'][i+8]['text']}", f"- 期望文档：{meta['cases'][i]['text']}", f"- 首版实际首位：{meta['cases'][predicted]['text']}", '- 新文本候选该查询已正确检索到期望文档。' if candidate_board.exists() and read(candidate_board).get('heldout',{}).get('indices',[None]*8)[i]==i else '- 新候选该查询尚无正确结果证据。', '']
lines += ['## 图片测试样例', '', '金门大桥：', '', '![金门大桥](samples/GoldenGate.png)', '', '厨房油画：', '', '![厨房油画](samples/kitchen_painting.jpg)', '',
          '图片来源与哈希：samples/manifest.json。上述视觉余弦比较的是视觉核心中间特征，不代表图片问答或生成式 LLM 能力。', '',
          '## 原始证据', '', '- 首版板端：bpu/results/text_runtime/、vision_runtime/，含输出 dumps 与 perf 日志。',
          '- 新候选：bpu/results/*_mixed_precision_validation.json；日志 bpu/logs/quantization_diagnosis/。',
          '- 软件敏感性实验：QUANTIZATION_DIAGNOSTIC_DATA.md、bpu/results/quantization_ablation.csv。',
          '- 后台编译/板端监督：bpu/logs/quantization_diagnosis/mixed_supervisor.json。', '',
          '尚未验收：完整多模态端到端、音频/视频、长期稳定性。未进行剪枝；未发布修复成功教程或 skill。']
(root/'TEST_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(results/'test_results_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(root/'TEST_RESULTS.md')
