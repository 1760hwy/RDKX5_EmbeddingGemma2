# 测量证据怎么读

`measured_metrics.json`是汇总口径；`release_acceptance.json`给出有限工程验收范围，不能代替原始数据。

- `batch_bpu_summary.json`：各轮/各批完整时间、相似度、检索与温频RSS观察。
- `batch_cpu_summary.json`：同六图调度的FP32四线程对照，小样本，非CPU100。
- `batch_bpu_pull_verification.json`：历史拉回319文件的SHA检查；其中313个NPY在本包 `raw/batch_bpu/`。
- `batch_cpu_pull_verification.json`：CPU历史拉回检查；原向量在 `raw/batch_cpu/`。
- `full_board_gate_summary.json`及 `raw/full_board_gate/`：完整板端语义和核心阶段。
- `web_api_checks.json`：四种真实API输入检查与错误请求恢复的历史记录。
- `pipeline_manifest.json`：精确候选路径、SHA与输入参考身份，不含对应二进制。

部分summary中包含原工程相对路径。本包仅重排证据文件的位置，运行代码仍要求ARTIFACTS.md列出的原运行路径。Windows私有路径和LAN地址在文本副本中脱敏，SHA清单同时记录原文本SHA与整理副本SHA；数值向量不修改。

这是已保存测量的归档，没有重新连板复测。原始summary的验收字段保持历史语义；工程范围报告单列，不把历史字段改成成功。

读取原向量示例（不做模型推理）：

```python
from pathlib import Path
import numpy as np
paths = sorted(Path('evidence/raw/batch_bpu').glob('*.npy'))
print('outputs:', len(paths))  # 此包应为313
v = np.load(paths[0])
print(v.shape, np.isfinite(v).all(), np.linalg.norm(v))
```

数据来自六图小样本轮转和短文本，四个猫帧相关；向量数量不能解释成独立场景数量。
