<p align="center">
  <strong>NexusQuant</strong>
</p>
<p align="center">
  将大语言模型的 KV 缓存压缩 10-33 倍。无需训练，一行代码搞定。
</p>
<p align="center">
  <a href="https://pypi.org/project/nexusquant-kv/"><img src="https://img.shields.io/pypi/v/nexusquant-kv?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/jagmarques/nexusquant/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/jagmarques/nexusquant"><img src="https://img.shields.io/github/stars/jagmarques/nexusquant?style=social" alt="Stars"></a>
</p>

---

Token 淘汰 + E8 格量化，在预填充后执行一次。无需训练、无需校准数据、无需修改模型。

## 安装

```bash
pip install nexusquant-kv
pip install "nexusquant-kv[hf]"  # 包含 HuggingFace transformers
```

## 快速开始

```python
from nexusquant import nexusquant_evict

with nexusquant_evict(model, quality="balanced"):
    output = model.generate(input_ids, max_new_tokens=512)
```

## 质量预设

在 Mistral-7B、A100、FP16 上测量。压缩比包含所有开销（缩放因子、索引、元数据）。

| 预设 | 压缩比 | PPL 退化 | 80 GB 上的上下文长度 |
|---|---|---|---|
| `high` | 10x | +0.4% | ~130 万 tokens |
| `balanced` | 17x | +1.3% | ~220 万 tokens |
| `max` | 33x | +2.6% | ~420 万 tokens |

已在 Mistral-7B、TinyLlama-1.1B、Llama-3-8B 上，针对学术、技术和创意文本进行验证。

## 工作原理

1. **重要性评分** — 通过跨头注意力权重（key-key 点积）对 token 进行排名
2. **Token 淘汰** — 丢弃得分最低的 token；始终保留 BOS 和最近的滑动窗口
3. **RoPE 去除** — 撤销 key 上的旋转位置编码，使其共享公共子空间，将量化误差降低约 0.7 个百分点
4. **Hadamard 旋转** — 将能量均匀分散到各维度，避免单个离群值主导量化尺度
5. **E8 格量化** — 将 8 个浮点数一组量化到 E8 根格（8 维空间中最密集的球填充），2 bits/dim
6. **Delta 编码 + zstd** — 相邻 token 产生相似的格索引；存储差值再用 zstd 压缩索引流，可再获得 2-3 倍压缩

Token 淘汰降低*数量*（60% 淘汰率下 2.5 倍）。E8 量化降低*精度*（熵编码后约 7 倍）。合计：17 倍。

## 与竞品对比

| 方法 | 压缩比 | PPL 退化 | 是否需要训练 |
|---|---|---|---|
| **NexusQuant** | **10-33x** | **+0.4-2.6%** | **否** |
| TurboQuant（Google） | ~5-6x | ~0% | 否 |
| KVTC（NVIDIA） | 最高 20x | <1% | 是（校准，约 10 分钟） |
| CommVQ（Apple） | ~8x | ~0% | 是（完整重训练） |
| Palu | 11x | ~25% rel | 是（校准） |

NexusQuant 是压缩比最高的免训练方法。KVTC 可实现相当的压缩比，质量更好，但需要校准数据。竞品数据来自其发表的论文，未在我们的硬件上复现。

## 许可证

Apache 2.0。详见 [LICENSE](LICENSE)。
