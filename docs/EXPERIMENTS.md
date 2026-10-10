# 量化源码的执行顺序与依赖

这是原实验程序的整理副本。多数输出目录固定，不要在唯一原始工程中无检查重跑，不要覆盖失败 attempt。

| 阶段 | 入口 | 依赖/进入条件 |
|---|---|---|
| FP32与导出 | `infer.py`, `benchmark.py`, `export_bpu_vision.py`, `export_bpu_fusion.py` | 官方权重、适配导出环境、厂商SDK |
| 原混合候选 | `quantize_trim_vision.py`, `quantize_bpu_fusion.py` | 固定shape的ONNX、校准数据、FP32参考；检查原精度配置 |
| 两层MLP诊断 | `probe_mlp_quant_boundary.py --explicit-shapes` | 原576量化、旧两层分段、像素/边界参考 |
| 完整MLP候选 | 同脚本 `--explicit-shapes --full-model` | 前项exit0及gate、原校准/quant、同fusion及12例FP32最终参考 |
| 首块编译 | `compile_mlp_boundary_chunk0.py --chunk 0` | 完整候选joint gate、权重/Softmax审计、六图slice等价、厂商HybridBuilder |
| 首块板端 | `test_mlp_boundary_chunk0_board.py` | 同SHA bin、独立加载、shape、同有效输入与新软件边界 |
| 其余块 | `prepare_mlp_remaining_chunks.py`, `supervise_mlp_remaining.py` | 首块真实板端通过；八块节点/张量字节与全latent等价；监督有限无盲重试 |
| 新板端全链 | `prepare_mlp_board_gate.py`, `validate_mlp_multimodal_board.py` | 八块+fusion部署SHA、六图latents、固定文本和FP32参考 |
| CPU/BPU100 | `prepare_mlp_pipeline_manifest.py`, `supervise_mlp_batch_acceptance.py` | 全链joint gate和核心100；同官方预处理/输入/候选；CPU先结束再BPU |
| 单张诊断 | `prepare_mlp_single_probe.py`, `supervise_mlp_single_probe.py` | 同已测candidate；小样本，不能充当100稳定性 |
| 汇总 | `report_mlp_acceptance.py` | 原summary、逐轮输出、SHA拉回报告与CPU同配置对照 |

Python实验多使用容器 `/work`。不要把这些路径当成 Windows 的可直接运行命令。厂商 Python 与 `/opt/embeddinggemma2-export` 导出环境不同；具体 imports 决定执行环境。

首块通过后编译其余块并不能跳过新全链精度和端到端速度。只更换某块就成为新候选，必须重新锁定清单；不得拼接旧性能与新精度。

SOURCE_INDEX.md 列出全部复制源码。源文件中读取哪些 `bpu/...` 文件，是具体依赖的最终依据；本表仅给阶段边界。原有自动监督程序不是通用安全构建工具，使用前必须核查固定输出路径、超时和本机工具链。
