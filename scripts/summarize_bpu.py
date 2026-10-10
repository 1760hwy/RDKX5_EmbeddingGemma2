"""Render current measured BPU results without turning pending jobs into success."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
bpu=root/'bpu'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None
state=read(bpu/'results/configured_supervisor.json') or {}
lines=['# EmbeddingGemma 2 / RDK X5 BPU 实验结果','',
    f"状态快照：{state.get('timestamp','未知')}。运行状态以原始退出码和板端 JSON 为准。",'',
    '采用原模型权重；文本固定 32 tokens，视觉 max_soft_tokens=70、固定方形网格。文本词表查找、视觉预处理及最终多模态文本网络不属于对应编译核心。','',
    '| 测试 | CPU FP32 | 编译模型板端 | 核心加速比 |','|---|---:|---:|---:|']
for scope,label,baseline in [('text','32-token 文本核心','cpu_text_core_l32.json'),
    ('vision','70 soft tokens 视觉核心','cpu_vision_core_s70.json')]:
    cpu=read(bpu/'results'/baseline)
    runtime=read(bpu/'results'/f'{scope}_runtime/summary.json')
    cpu_ms=cpu['median_ms'] if cpu and cpu.get('status')=='complete' else None
    bpu_ms=runtime['average_core_ms'] if runtime and runtime.get('status')=='complete' else None
    lines.append(f"| {label} | {cpu_ms:.3f} ms" if cpu_ms is not None else f'| {label} | 未完成')
    lines[-1]+=(f' | {bpu_ms:.3f} ms | {cpu_ms/bpu_ms:.2f}× |' if bpu_ms and cpu_ms else ' | 未完成 | 未确认 |')
lines+=['','CPU 为重复测量中位数；编译模型为 hrt_model_exec 单线程 100 帧平均值，包含编译核心内部 CPU/BPU 执行与调度。两者都排除模型加载和核心外预处理。运行时 perf 使用其生成的输入，精度另用实际固定样例验证。','']
for scope in ['text','vision']:
    lines += [f'## {"文本" if scope=="text" else "视觉"}', '']
    gate=state.get('scopes',{}).get(scope,{})
    current_runtime=read(bpu/f'results/{scope}_runtime/summary.json')
    if current_runtime and current_runtime.get('status')=='complete':
        lines.append('板端测试完成，以本次结果 JSON 为准；监督脚本的旧失败快照已被后续测试覆盖。')
    else:
        lines.append(f"监督状态：`{gate.get('gate','未知')}`。")
    log=bpu/f'logs/{scope}_configured_bounded/compile.log'
    if log.exists():
        content=log.read_text(errors='replace')
        if 'The main quantized node information:' in content:
            table=content.split('The main quantized node information:',1)[1].split('The quantized model output:',1)[0]
            rows=[line for line in table.splitlines() if 'BPU' in line.split() or 'CPU' in line.split()]
            cpu_rows=sum('CPU' in line.split() for line in rows)
            softmax_rows=[line for line in rows if 'QUANTIZED_SOFTMAX' in line]
            softmax_cpu=sum('CPU' in line.split() for line in softmax_rows)
            lines.append(f'编译日志算子表识别 {len(rows)} 行，其中 CPU {cpu_rows} 行；Softmax 分解节点 {len(softmax_rows)} 行，其中 CPU {softmax_cpu} 行。这是静态算子记录，不能当成实际切换次数或全网络纯 BPU 的证明。')
    runtime=read(bpu/f'results/{scope}_runtime/summary.json')
    if runtime and runtime.get('status')=='complete':
        lines.append(f"同校准样例的最小输出余弦相似度：{runtime['minimum_cosine']:.6f}。")
        if scope=='text':
            lines.append(f"校准样例检索 Top1：{runtime['retrieval_top1']}/{runtime['retrieval_total']}。")
            held=runtime.get('heldout')
            if held:
                lines.append(f"未参与校准的人工中英检索集：浮点 {held['float_top1']}/{held['total']}，量化 {held['top1']}/{held['total']}；最小向量余弦 {held['minimum_cosine']:.6f}。不代表真实业务准确率。")
        cpu=read(bpu/'results'/('cpu_text_core_l32.json' if scope=='text' else 'cpu_vision_core_s70.json'))
        if cpu and cpu.get('status')=='complete':
            speedup=cpu['median_ms']/runtime['average_core_ms']
            quality=runtime['minimum_cosine']>=0.95
            if scope=='text':
                held=runtime.get('heldout')
                quality=quality and bool(held) and held['minimum_cosine']>=0.95 and held['top1']>=held['float_top1']
            if speedup>=2 and quality:
                lines.append('当前核心候选达到至少 2× 加速且小样本方向相似度/检索保持的初步筛选门槛，有继续做应用级测试的价值；尚不能称为业务验收通过。')
            elif not quality:
                lines.append('当前量化候选存在明显小样本精度损失，不建议按此配置用于应用；需要先修复精度，再评价性能收益。')
            else:
                lines.append('当前核心未达到本实验至少 2× 加速的初步筛选门槛，不能依据峰值 TOPS 宣称有明显收益。')
    elif runtime:
        lines.append(f"板端失败：`{runtime.get('error','请查看原始日志')}`。")
    lines.append('')
lines+=['## Softmax 与已处理问题','',
    '当前正式配置显式启用 `optimization: set_Softmax_input_int8`。独立 `[1,4,32,32]` Softmax 已编译为 BPU INT8，板端 200 帧平均 0.369180 ms，输出余弦 0.99973476。这只能证明独立算子路径。','',
    '原始 padding mask 的 -1e9 使首层 INT8 校准步长达到 7874015.5，相关转换主动中止并保留。当前导出只把 mask 常量改为 -32；原权重保持原样。16 个校准文本、16 个不参与校准的文本和 2 张图片均通过浮点数值对比。该近似仍需在更多真实输入上检查。','',
    '第一次默认 checker 的并行任务共用临时目录，也已停止并保留证据；当前各任务目录独立。','',
    '完整模型尚未完成板端验证时，不能判断 10 TOPS 对本模型的实际收益。视觉候选还不包含最终图像 embedding 网络，不能把视觉核心延迟当作完整图文检索延迟。','',
    '官方依据：[PTQ 编译参数及 Softmax INT8 开关](https://developer.d-robotics.cc/oe_x5_doc/cn/oe_mapper/source/ptq/ptq_tool/hb_mapper/hb_mapper_makertbin.html)。']
(root/'BPU_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(root/'BPU_REPORT.md')
