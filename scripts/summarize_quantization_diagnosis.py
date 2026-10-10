"""Publish measured diagnostic results, preserving pending/failed statuses."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
results=root/"bpu/results"
lines=["# 量化精度诊断数据", "", "由原始 JSON 生成。所有 ablation 均为单个校准样例的软件敏感性测试，不能当作板端修复或业务精度。", "",
       "| 核心 | 对照实验 | 状态 | 参考向量余弦 |", "|---|---|---|---:|"]
rows=[]
for scope in ["text","vision"]:
    for experiment in ["softmax","normalization","weight_feature","precision_sweep","range_sweep","softmax_range"]:
        path=results/f"{scope}_{experiment}_ablation.json"
        if not path.exists():
            lines.append(f"| {scope} | {experiment} | 未产生结果 | — |")
            continue
        report=json.loads(path.read_text())
        for row in report.get("experiments",[]):
            lines.append(f"| {scope} | {row['mode']} | 已测，任务 {report['status']} | {row['reference_cosine']:.6f} |")
            rows.append({"scope":scope,"experiment":row["mode"],"reference_cosine":row["reference_cosine"],"relative_rmse":row["relative_rmse"],"seconds":row["seconds"],"changed_count":row["changed_count"]})
        if report["status"]!="complete":
            lines.append(f"| {scope} | {experiment} 剩余项 | {report['status']} | — |")
    tracepath=results/f"{scope}_calibration_trace.json"
    if tracepath.exists():
        trace=json.loads(tracepath.read_text())
        invalid=[row for row in trace.get("checks",[]) if row.get("quant_nonfinite_fraction",0)>0]
        lines += ["", f"{scope} 逐层 trace：{trace['status']}。所选 {len(trace.get('checks',[]))} 个中间输出中，有 {len(invalid)} 个出现伪量化非有限值。可能包含 padding，仍需有效 token 范围与实际量化图交叉验证。", ""]
lines += ["", "## 厂商实际转换后的混合精度候选", "", "这是量化 ONNX 的软件验证，不是板端测试。设定 A16/W8/Softmax8，厂商可能回退部分输入到 INT8，应结合转换日志与后续 placement 判断。", "",
          "| 核心 | 状态 | 已验证样例 | 最低余弦 | 数值门槛 |", "|---|---|---:|---:|---|"]
for scope in ["text","vision"]:
    path=results/f"{scope}_mixed_precision_validation.json"
    if path.exists():
        report=json.loads(path.read_text())
        checks=report.get("checks",[])
        minimum=f"{min(row['cosine'] for row in checks):.6f}" if checks else "—"
        gate=str(report.get("numerical_gate_passed","尚未判定"))
        lines.append(f"| {scope} | {report['status']} | {len(checks)} | {minimum} | {gate} |")
lines += ["", "三阶段对比：优化浮点输出保持参考；量化 ONNX 在当前两图和两个文本样例上与板端逐元素一致。原始记录 `bpu/results/*_quantization_diagnosis.json`。", "",
          "先验证软件修复候选，再编译与板端验证。整体验收尚未通过；文本板端候选的最新精度和速度见 TEST_RESULTS.md。"]
(root/"QUANTIZATION_DIAGNOSTIC_DATA.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
with (results/"quantization_ablation.csv").open("w",newline="",encoding="utf-8") as stream:
    writer=csv.DictWriter(stream,fieldnames=["scope","experiment","reference_cosine","relative_rmse","seconds","changed_count"])
    writer.writeheader()
    writer.writerows(rows)
print(root/"QUANTIZATION_DIAGNOSTIC_DATA.md")
