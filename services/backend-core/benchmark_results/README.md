# Benchmark 结果索引

> 所有结果均由仓库内脚本可复现。每份 JSON 都记录了完整 `config`，引用数字时请连同 `n` 一起写。

## 运行环境（复现前必读）

从宿主机（非容器）运行脚本时，必须设置以下环境变量，否则会**静默产生错误结果**：

```bash
export no_proxy=localhost,127.0.0.1,::1        # 系统代理会劫持 localhost → httpx 502
export NO_PROXY=localhost,127.0.0.1,::1
export HF_HUB_OFFLINE=1                        # 否则 pytest/脚本挂在 HF 联网检查
export TRANSFORMERS_OFFLINE=1
```

不设 `no_proxy` 时，`VectorIndex` 会**静默降级到空的内存索引**，检索结果为空但流程照常跑完 —— 数字会是假的。脚本已对这种情况硬失败。

## 检索策略对比 (`strategy_comparison_*.json`)

```bash
cd services/backend-core
python3 scripts/run_strategy_comparison.py --dataset expanded \
    --strategies direct,hybrid,rewrite --output benchmark_results
# 测检索缓存：加 --repeat 2
```

| 文件 | n | 策略 | 说明 |
|---|---|---|---|
| `strategy_comparison_20260920T203628Z.json` | 20 | direct, hybrid | 首次跑通（含去重 bug 修复前的异常 recall） |
| `strategy_comparison_20260921T040602Z.json` | 200 | direct, hybrid | 去重 bug 修复后（**缓存口径不对称**，仅留档） |
| `strategy_comparison_20260921T050008Z.json` | 20 | + rewrite | 首次暴露 rewrite 负收益 |
| `strategy_comparison_20260921T051401Z.json` | 200 | + rewrite | 三策略完整对比（缓存口径不对称，仅留档） |
| `strategy_comparison_20260921T083648Z.json` | 200 | direct, hybrid | **公平延迟口径（`use_cache=False`）——引用延迟时用这份** |

**n=200 实测结论**（三条策略统一 `use_cache=False`，避免 hybrid 因命中检索缓存而显得更快）：

| 策略 | MRR | R@5 | HR@5 | 平均延迟 |
|---|---|---|---|---|
| direct（纯向量） | 0.8454 | 0.9000 | 0.955 | 33.0 ms |
| hybrid（向量+KG+RRF） | **0.8569** | **0.9054** | 0.955 | **30.3 ms** |
| rewrite（LLM 改写后 hybrid） | 0.8029 | 0.9054 | 0.95 | 1822.9 ms |

- **口径修订**：早期报告曾称 hybrid 延迟为 direct 的 ×0.64。复核发现 `direct` 直调 `_chroma_search`（绕过缓存），而 `hybrid` 走 `search()` 并命中缓存——该优势部分是缓存假象。公平对比为 **×0.92**。
- **rewrite 是负收益**：MRR −0.054（相对 hybrid），延迟 ×60；200/200 改写全部成功。
  因此 `rag_rewrite_enabled` 默认 `False`。**这是如实报告的负结果，未调 prompt 凑数字。**

## 意图识别评测 (`intent_eval_*.json`)

```bash
cd services/backend-core
python3 scripts/run_intent_eval.py --no-cache --output benchmark_results   # 冷启动（头条）
python3 scripts/run_intent_eval.py --output benchmark_results              # 含缓存命中
```

评测集：`src/rag/evaluation/datasets/intent_labeled.json`（72 条人工标注，含刻意难例）。

| 文件 | n | 准确率 | cold |
|---|---|---|---|
| `intent_eval_20260921T042004Z.json` | 72 | 0.7500 | ✅ 修复前基线 |
| `intent_eval_20260921T042250Z.json` | 72 | 0.7500 | ❌ 含缓存 |
| `intent_eval_20260921T045655Z.json` | 72 | **0.9306** | ✅ 修复后 |

**结论**：冷启动准确率 **75.0% → 93.1%**。18 个初始错判**全部**来自规则快路径；修复后规则路径覆盖 95.8% 的消息且**零模型调用**，融合路径只处理 4.2% 且准确率 100%。
