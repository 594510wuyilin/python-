# 项目一：基于 NumPy 的以图搜图（OpenCLIP ViT-B/32 图像编码器）

本项目仅使用纯 NumPy 复现 OpenCLIP ViT-B/32 图像编码器的完整前向推理流程。加载课程提供的 152 个预训练 `float32` 视觉权重张量后，可在 CPU 上将输入图像批处理编码为 512 维 L2 单位特征向量，并与独立 FP32 参考输出保持数值一致。

## 项目功能

- **图像切块（`ImageEncoder.patchify`）**：将预处理后的 `[B, 3, 224, 224]` 图像批次按行优先顺序切分为 `7 × 7 = 49` 个 `32 × 32` RGB 图块，块内保持 `(C, H, W)` 顺序展平为 `[B, 49, 3072]`。
- **词元嵌入与位置编码（`ImageEncoder.embed`）**：对图块执行无偏置线性投影（`3072 -> 768`），在首行拼接可学习 `CLS` 词元并广播叠加位置编码，输出 `[B, 50, 768]`。
- **多头自注意力（`MultiHeadAttention`）**：支持 12 头并行缩放点积自注意力（每头 64 维）、可选因果掩码（`causal=True` 兼容期末文本编码器）以及多头合并与输出投影。
- **残差前馈块（`TransformerBlock`）**：串联 12 层独立权重的 Pre-LN Transformer 块，包含两次残差连接与 `768 -> 3072 -> 768` 精确 GELU 前馈网络。
- **特征读出与归一化（`ImageEncoder.__call__`）**：提取末层 `CLS` 词元，经 `post_norm`、线性投影（`768 -> 512`）与 L2 单位化输出最终图像语义向量 `[B, 512]`。

## 仓库与代码结构

本仓库仅提交个人实现代码、依赖清单与说明文档，不包含体积较大的课程预训练权重与参考数据。与课程附件 `project1-学生版.zip` 结合后的完整目录结构如下：

```text
project1-numpy-student/
├── image_encoder.py          # [本仓库核心实现] MultiHeadAttention / TransformerBlock / ImageEncoder
├── README.md                 # [本仓库提供] 项目功能、结构与复现运行说明
├── requirements.txt          # [本仓库提供] Python 第三方依赖清单 (NumPy / Pillow / imageio)
├── helpers.py                # [课程附件提供] LayerNorm、精确 erf GELU、Softmax、causal_mask、unit_vector
├── preprocess_image.py       # [课程附件提供] 短边 224 缩放、中心裁剪与通道标准化
├── runtime.py                # [课程附件提供] 配置、权重与图像加载工具
├── check.py                  # [课程附件提供] 分阶段数值精度验收脚本
├── run.py                    # [课程附件提供] 命令行端到端特征提取脚本
├── config.json               # [课程附件提供] ViT-B/32 模型结构与标准化常量配置
├── weights/
│   └── image_encoder.npz     # [课程附件提供] 152 个预训练 float32 视觉权重张量（不上传至 GitHub）
└── assets/
    ├── images/               # [课程附件提供] 10 张样例测试图像
    └── reference_outputs.npz # [课程附件提供] 独立 FP32 分阶段参考输出（不上传至 GitHub）
```

## 环境依赖与数据获取

- **运行环境**：Python 3.9+（实测通过环境：Python 3.14.0、NumPy 2.3.4、Pillow 12.2.0，纯 CPU 运行，无需 PyTorch 或 GPU）。
- **依赖安装**：见 `requirements.txt`（包含 `numpy>=1.24,<3`、`Pillow>=9.2,<13`、`imageio>=2.31,<3`）。
- **权重与样例数据获取**：预训练权重 `weights/image_encoder.npz` 及测试图像位于深圳大学畅课平台课程附件 `project1-学生版.zip` 中，解压后按下方步骤替换 `image_encoder.py` 即可运行。

## 运行与验收步骤

1. 解压课程附件 `project1-学生版.zip`，将本仓库的 `image_encoder.py` 复制到 `project1-numpy-student` 目录并替换同名起始文件。
2. 在 `project1-numpy-student` 目录下打开终端，依次执行：

```bash
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

- `check.py --stage all`：依次校验 152 个权重张量完整性及 `preprocess`、`patches`、`tokens`、`block0`、`last_tokens`、`output`、`batch` 七项数值精度（实测最终 `output` 最大绝对误差 `5.439e-07`，远低于 `2e-5` 容差）。
- `run.py --output output_tokens.npz`：对 10 张样例图像提取特征并保存至 `output_tokens.npz`（含 `ids` 与形状为 `(10, 512)` 的 `output_tokens`）。
