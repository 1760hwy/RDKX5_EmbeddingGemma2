# 外部产物与复现边界

准备目录不包含权重、ONNX、HBIR/HBM、bin、厂商运行库、wheel、虚拟环境或原始媒体。

## CPU 路线

1. 合法获取官方 `google/embeddinggemma-2` 快照；记录 revision 与各文件 SHA。
2. 放入项目根 `model/`，保留官方目录结构与全部配置/分词器文件。
3. 安装适配 ARM64/Python 3.10 的已验证依赖；先 `pip check` 再导入模型类。
4. `mkdir -p results` 后运行 `scripts/infer.py`。

## BPU 网页路线

从已有验收工程按原相对路径提供以下文件。没有可分发的模型下载链接；不得把本清单误认为二进制已经包含在源码包。

- `model/`：本地官方快照，用于官方处理器与 CPU embedding lookup。
- `bpu/runtime/libembedding_bpu.so`：板端源码构建的适配库；厂商头文件/运行库由合法 SDK 提供。
- `bpu/vision_trim576/mlp_boundary_full16/full_board_inputs/pipeline_manifest.json`：运行清单。准备包副本位于 `evidence/pipeline_manifest.json`，部署时恢复到原相对路径。
- 八块视觉模型与一个融合模型：精确路径和 SHA 见 `evidence/measured_metrics.json` 的 `candidate`；融合也可查 `evidence/pipeline_manifest.json` 的 `gate_manifest.fusion`。
- `samples/GoldenGate.png`、`samples/kitchen_painting.jpg`、`samples/prepared/video_frames/frame_01.png`：网页固定示例与启动网格参考，来源见第三方说明。即使只测试上传图片，当前引擎初始化仍需要 GoldenGate 参考图。

以上匹配后执行预检，再做板端加载与真实请求。运行库存在不等于依赖完整；可以用 `ldd bpu/runtime/libembedding_bpu.so` 观察缺失库，但不能随意替换 ABI。

## 完整量化重建

原浮点 ONNX、混合校准模型、校准像素、FP32 参考、历史两层诊断报告、首块板端诊断、拆分等价输出和厂商缓存属于独立实验依赖。未随包分发，研究者需要按实验源码的依赖自行合法生成。详见 EXPERIMENTS.md。

当前不能承诺一个陌生读者仅用此包就能重建相同 SHA 的所有 bin。正式 release 应增加干净目录构建入口、数据获取与许可说明、版本锁定、独立复现记录，才可标为完整可复现发行版。
