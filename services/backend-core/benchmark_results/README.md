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

## 检索缓存收益 (`--repeat 2`)

`--repeat 2` 会把同一评测集跑两遍并分别报告延迟。此模式下 hybrid 腿**允许命中检索结果缓存**
（`use_cache=True`），第二遍即缓存命中率的上界；`direct` 直调 `_chroma_search` 不经缓存，
因此作为对照组，用来区分"缓存收益"与"进程预热收益"。

```bash
python3 scripts/run_strategy_comparison.py --dataset expanded \
    --strategies direct,hybrid --repeat 2 --output benchmark_results
```

n=200 实测（`strategy_comparison_20260921T090049Z.json`）：

| 策略 | 第 1 轮 | 第 2 轮 | 倍数 | 说明 |
|---|---|---|---|---|
| direct（不经缓存，对照组） | 31.81 ms | 14.46 ms | ×2.20 | 纯进程预热（jieba/OS 缓存），与检索缓存无关 |
| hybrid（经缓存） | 16.75 ms | **0.01 ms** | **×1675** | 第 2 轮完全命中缓存，未执行任何检索 |

**结论**：检索结果缓存在重复查询上把延迟从 ~17ms 降到 ~0.01ms（同进程内），
即**重复查询不再触发向量检索与图谱查询**。对照组证明该收益不是预热假象。
缓存 TTL 默认 300s（`rag_cache_ttl_seconds`），一致性由 TTL 兜底。

| 文件 | n | 策略 | 说明 |
|---|---|---|---|
| `strategy_comparison_20260920T203628Z.json` | 20 | direct, hybrid | 首次跑通（含去重 bug 修复前的异常 recall） |
| `strategy_comparison_20260921T040602Z.json` | 200 | direct, hybrid | 去重 bug 修复后（**缓存口径不对称**，仅留档） |
| `strategy_comparison_20260921T050008Z.json` | 20 | + rewrite | 首次暴露 rewrite 负收益 |
| `strategy_comparison_20260921T051401Z.json` | 200 | + rewrite | 三策略完整对比（缓存口径不对称，仅留档） |
| `strategy_comparison_20260921T083648Z.json` | 200 | direct, hybrid | **公平延迟口径（`use_cache=False`）——引用延迟时用这份** |
| `strategy_comparison_20260921T090049Z.json` | 200 | direct, hybrid | **`--repeat 2` 缓存收益测量（见下节）** |

**n=200 实测结论**（三条策略统一 `use_cache=False`，避免 hybrid 因命中检索缓存而显得更快）：

| 策略 | MRR | R@5 | HR@5 | 平均延迟 |
|---|---|---|---|---|
| direct（纯向量） | 0.8454 | 0.9000 | 0.955 | 33.0 ms |
| hybrid（向量+KG+RRF） | **0.8569** | **0.9054** | 0.955 | **30.3 ms** |
| rewrite（LLM 改写后 hybrid） | 0.8029 | 0.9054 | 0.95 | 1822.9 ms |

- **口径修订**：早期报告曾称 hybrid 延迟为 direct 的 ×0.64。复核发现 `direct` 直调 `_chroma_search`（绕过缓存），而 `hybrid` 走 `search()` 并命中缓存——该优势部分是缓存假象。公平对比为 **×0.92**。
- **rewrite 是负收益**：MRR −0.054（相对 hybrid），延迟 ×60；200/200 改写全部成功。
  因此 `rag_rewrite_enabled` 默认 `False`。**这是如实报告的负结果，未调 prompt 凑数字。**

## 融合策略消融 (`fusion_ablation_*.json`)

```bash
cd services/backend-core
python3 scripts/run_fusion_ablation.py --dataset expanded --course cs201 --output benchmark_results
python3 scripts/run_fusion_ablation.py --list-courses     # 磁盘上有哪些语料
```

**⚠️ 语料复核状态：cs301 的标注未经人工核验，其数字为初步结果。**

| 文件 | 语料 | 标注 |
|---|---|---|
| `expanded_queries.json` | cs201，22 KP | 人工 + 自动生成（原始） |
| `expanded_queries_cs301.json` | cs301，10 KP | **机器起草，尚未人工复核** |

**cs301 那一行是 `unreviewed` 的，读下面结论时请连同这一点一起读。** 标注决定 MRR 的分子
（`relevant_kp_ids` 错一条，MRR 就跟着动），本题的"反转"完全可能因标注修正而消失或翻转。
复核方法见该文件 `_meta.review_note`。**在复核完成前，任何依赖 cs301 数字的决策都应推迟。**

同一份代码、同一个会话内实测（cs301 待复核）：

| 语料 | n | rrf | minmax | score | 最优 |
|---|---|---|---|---|---|
| cs201（数据结构与算法） | 200 | 0.8370 | 0.8197 | **0.8851** | **score** |
| cs301（操作系统）⚠️未复核 | 40 | 0.9542 | **0.9875** | 0.9750 | **minmax** |

- **已确证的部分**：`score` 的优势**不是跨语料普适的**——cs201 上它领先 `minmax` 0.065，
  而在另一份语料上两者次序变化。这一点即便 cs301 需修正也成立：**不存在无条件的默认最优**。
- **待复核的部分**：cs301 上"`minmax` 反超 `score`"的具体幅度，以及 `rrf` 的名次。
- 因此 `hybrid_fusion_method` 的默认值保留 `score`（cs201 是主课程），但**不再是"最优策略"的
  声明，而是"在 cs201 上的选择"**。`src/config.py` 的注释已同步写明这一边界。
- **量纲假设被削弱**：当初选 `score` 的理由之一是"按原始分数排序即可"，而 cs301 上
  `minmax`（显式做源内归一化）表现更好——与 issue #3 的担忧一致：ChromaDB 相似度
  （`1 - cosine_distance`）与 Neo4j 离散匹配质量量纲不同。

**语料来源**：cs301 由 `scripts/seed_cs301.py` 播种知识图谱（10 KP）后编写；
入库需先 `seed_embeddings.py`（注意该脚本默认 compose 主机名，见 `seed_cs301.py` 文档）。

**记录口径**：`cache_bypassed: true`（检索缓存全部绕过）。2026-09-26 起的结果含
`course_id` 字段，**推断语料请以该字段为准**（文件名不编码课程）；更早的 artifact
（含做出 `score` 默认决策所依据的 `fusion_ablation_expanded_20260923T130739Z.json`）
没有该字段，其 `course_id` 只能从当时的单课程事实推定为 cs201。
早期（2026-09-23）的 `fusion_ablation_expanded_20260923T130739Z.json` 是 cs201、n=200，
即当初做出 `score` 默认决策所依据的那份。

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
