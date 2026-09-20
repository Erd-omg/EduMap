# 亮点三：RAG 混合检索 + 语义分块 + Cross-Encoder 重排序

> **核心数字（口径详见 §4）**：MRR 0.841 / HitRate@3 0.930 / P@1 0.755（n=200 扩展集，**未启用重排**）｜全管线 P50 19.4ms / P95 54.7ms（100 次压测，未启用重排）
>
> **诚实前置（重要）**：仓库内 5 份基准报告（`services/backend-core/benchmark_results/`）的 `config.reranker_enabled` **全部为 `false`**。Cross-Encoder 重排是**已实现、默认关闭**的可选项，它与基线的对照实验**尚未做过**。因此面试时**不要说**"加了精排之后 MRR 达到 0.841"——这是错误归因；正确表述见 §4。

## 1. 背景阐述

接下来讲 RAG，这是面试官一定会深挖的部分。我们这个系统里有个 Mentor Agent，负责回答学生关于课程内容的提问，比如"数组的越界问题为什么危险"、"快排的最坏复杂度是怎么推导的"。它不能瞎编，答案必须基于我们库里已有的学习资料和知识图谱，还要带来源引用。

数据侧大概是这样的规模：一个课程的知识点图谱有几百到上千个节点（存在 Neo4j），学习资料被切成片段存进 ChromaDB，单个课程大概 **500~2000 个文本片段**。请求是**低并发高延迟**型的——单用户实时问答，但对**回答质量**要求极高，因为这是教育产品，答错了就是在教坏学生。

## 2. 问题剖析

虽然当时我们已经有一个能跑的检索了，但**第一版是纯 embedding 相似度检索**，用 BGE 的 bi-encoder 做向量召回。跑起来之后发现两个很恼火的问题：

第一个是**假阳性**。Bi-encoder 是"把整段话压成一个向量"，语义上相近但主题不同的内容，余弦相似度也会很高。举个真实例子，学生问"**排序算法的稳定性**"，检索回来一堆讲"**稳定婚姻匹配**"的片段——都带"稳定"两个字，但根本不是一回事。如果直接把这些塞进上下文喂给 LLM，答案就会被带偏。

第二个是**分块质量差**。最早用固定大小切块，一个知识点被拦腰切到两个 chunk 里，检索时只能召回一半，信息不完整，LLM 拿到的上下文是残缺的。

如果放任不管，最严重的后果就是**Mentor 给出"看起来有依据、实际上跑题"的错误答案，而且带着来源引用，学生会信以为真**。对教育产品来说，这是最高等级的事故，比不回答严重得多。

## 3. 方案构思与技术实现

我的解法是三段式：**混合召回 → 语义分块 → Cross-Encoder 精排**。

**第一段：双路召回。** 为什么用"Neo4j + ChromaDB 双路"而不是纯向量？因为有些问题本质是**事实查询**——"快排的前置知识点是什么"这种，图谱里的 `(A)-[:PREREQUISITE_OF]->(B)` 边（前置）与 `(A)-[:RELATED_TO]->(B)` 边（相关）就是精确答案，向量检索反而可能召回一堆"讨论快排的笔记"这种噪声。关系名以 `scripts/db/seed/neo4j-seed.cypher` 为准：图谱共 2 类节点（`Course` / `KnowledgePoint`）、3 类关系（`HAS_TOPIC` / `PREREQUISITE_OF` / `RELATED_TO`），**没有 `REQUIRES`**。所以双路召回：Neo4j 走 Cypher 精确匹配（按名字 `CONTAINS`、按关系遍历），ChromaDB 走语义相似度，最后按 `source_type:source_id` 去重合并。

**第二段：语义分块。** 我实现了两个分块器：递归分块器（按段落→句子→词的分隔符层级切）和**语义分块器**。语义分块器的思路是这样的——对相邻两个句子分别做 embedding，算余弦相似度，如果相似度骤降，说明这里发生了**主题切换**，就在这个位置切一刀：

```python
def _find_boundaries(self, embeddings, sentences):
    boundaries = [0]; running_chars = 0
    for i in range(len(embeddings) - 1):
        sim = cosine(embeddings[i], embeddings[i + 1])   # 相邻句相似度
        running_chars += len(sentences[i])

        topic_shift = sim < self._similarity_threshold   # 阈值 0.7
        exceeds_max = running_chars > self._max_chunk_size
        meets_min = running_chars >= self._min_chunk_size

        if (topic_shift and meets_min) or exceeds_max:
            boundaries.append(i + 1); running_chars = 0
    boundaries.append(len(sentences))
    return boundaries
```

**为什么选语义分块而不是纯固定大小？** 因为学习资料的知识密度不均匀——有的段落讲概念要一整段才完整，有的几句就换话题了。固定 512 字符一刀切，会把"完整的知识点"切碎。语义分块保证每个 chunk 内部是**语义内聚**的，检索召回的是"一段完整的意思"，而不是"半句话"。

**第三段：Cross-Encoder 重排序。** 这是解决假阳性的关键。Bi-encoder 是一次编码"查询和文档各自成向量再算相似度"，快但粗糙；**Cross-Encoder 是把"查询+文档"拼接成一个整体输入模型**，模型能真正看到两者的交互，准确率高一个档次，但代价是每对都要跑一次前向，慢。所以我的策略是：**bi-encoder 负责粗筛出 top-20 候选（快），cross-encoder 在这 20 个里精排出 top-5（准）**，两边都不耽误：

```python
async def rerank(self, query, results, top_k=5):
    if not results or self._model is None:
        return results[:top_k]          # 模型没加载 → 优雅降级
    pairs = [(query, r.content[:1536]) for r in results]  # 20 对
    scores = self._model.predict(pairs, batch_size=16)
    for i, score in enumerate(scores):
        # 兼容标量 logits 和 [neg, pos] 两种输出
        results[i].score = sigmoid(score) if isinstance(score, (int, float)) else score[1]
    return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]
```

**兜底细节**：重排序模型如果加载失败，`is_loaded` 是 False，直接按原始分数返回 top-K，**功能不降级**；预测抛异常也一样，try/except 兜住。选型上我对比过 BGE 的 reranker（准确率更高但模型 1.1GB，部署重）和 MiniLM cross-encoder（80MB，中文能力够用），MVP 阶段选了后者，把"**更换模型 = 改一行配置**"的能力留好。

## 4. 优化效果与压测数据

这一段的数字必须**分清"实测可复现"与"设计期估算"**两类，因为有离线评测框架撑着（亮点四会讲），实测部分可以直接 grep 到原始 JSON：

- **检索质量（实测，`benchmark_results/latest_expanded.json`，n=200 扩展集）**：
  - 配置：`BAAI/bge-small-zh-v1.5` 向量召回 + Neo4j 关键词召回，双路合并，**`reranker_enabled=false`**、未启用图谱查询扩展
  - MRR **0.841**｜HitRate@3 **0.930**｜HitRate@5 **0.960**｜P@1 **0.755**（四分之三的查询，第一名就是正确答案）
  - 知识点覆盖率 **100%**（22/22 个数据集知识点在 Neo4j 中均命中）
- **检索质量（实测，`benchmark_results/latest_eval.json`，n=20 冒烟集）**：
  - **同一配置**下 MRR **0.900**｜HitRate@3 **0.950**｜P@1 **0.850**，覆盖率 100%（17/17）
  - **口径提醒**：小集指标天然更高，引用时**必须写明是 n=20 还是 n=200**，混用会被面试官抓
- **检索延迟（实测，`latest_eval.json` → `latency_benchmark`，100 次压测）**：
  - 全管线（**未启用重排**）：min 13.6ms｜**P50 19.4ms**｜**P95 54.7ms**｜P99 96.9ms｜avg 25.8ms
- **重排的收益与代价（【估算】，未做对照实验）**：
  - 设计预期是"精排会让 P95 从 ~55ms 上到百毫秒级"，因此**默认关闭、延迟敏感路径不启用**
  - **推荐面试表述**："重排我实现了、一行配置能开，但**我没拿它跟基线做过 A/B，所以不给提升数字**"——这比编一个百分比可信得多，且与 `docs/保研材料/面试追问-QA清单.md` 的口径一致
- **分块质量（未做对照实验）**：语义分块在原理上保证 chunk 语义内聚，但**我没有单独跑过"语义分块 vs 固定分块"的 A/B**，面试中不要给具体提升值，只说设计动机与实现

> **本节修正说明**：早期版本称"加上 Cross-Encoder 精排之后（最优配置）MRR=0.841"，经核对原始报告为**错误归因**——该数字来自未启用重排的 n=200 基线；"n=20 标注集 HitRate@3 0.83→0.930"则把 n=200 的数字安到了 n=20 数据集上（n=20 实测为 0.950）。相关数字已被本节替换。

## 5. 复盘总结

一句话：**检索质量不是靠一个模型解决的，是"粗筛保召回、精排保精度、分块保上下文完整"三层配合的结果**。踩过最大的坑是刚开始迷信"换个大模型 embedding 就万事大吉"，实际上一提准召率就发现瓶颈在假阳性和分块碎化上——**先把管线每一环都变成可评测的，再谈优化**。

**但要如实区分"已完成"与"未验证"**：截至当前，双路召回 + 语义分块已落地并跑出基线（§4），**Cross-Encoder 精排只实现了、没做 A/B**，语义分块的独立收益也没单独测过。所以更准确的一句话是：**"三层的实现都在，但只有前两层的收益被数据验证过"**——面试时这样讲反而更可信，还能顺势说明下一步的验证计划。

**顺带一个真实的可测性问题（已修复）**：早期版本双路合并是**裸合并**——Neo4j 关键词命中的 score 恒为 `1.0`，而向量分数是 0~1 的余弦相似度，两者直接丢进同一个排序列表，**关键词命中会系统性压过向量结果**。这已经修掉了：

- **Neo4j 关键词打分**改为按匹配质量分档（exact 1.00 / prefix 0.85 / substring 0.70 / jaccard 0.50~0.95 / 兜底 0.40），消除恒为 1.0 的问题
- **双路融合**改为可配置，默认 **RRF（Reciprocal Rank Fusion，`score(d) = Σ 1/(k+rank)`，k=60）**，仅依赖排序、对量纲差异免疫；也支持 `minmax`（各源 min-max 归一化）和 `score`（旧行为，用于对照）
- 切换由 `settings.hybrid_fusion_method` 控制，可不改代码做消融对比

## 6. 常见面试题与追问点

**Q1：Cross-Encoder 和 Bi-Encoder 的本质区别？为什么不能只用 Cross-Encoder？**

A：Bi-Encoder 把查询和文档分别编码成向量，相似度是两个向量的余弦，**可以离线预计算**，百万级文档也扛得住，但"查询和文档的交互"被压缩成了一个向量点积，信息有损；Cross-Encoder 把两者拼接后一起过模型，能建模交互，准，但是**每来一个新查询都要把所有候选跑一遍前向**，不能离线。所以标准做法是**两段式**：bi-encoder 粗筛 + cross-encoder 精排，用"准"去校验"快"的 top-K。

**Q2：Cross-Encoder 的分数怎么跨模型归一化？不同模型的 logits 分布不一样，怎么设阈值？**

A：这是个好问题。我们一般不直接对 raw score 设绝对阈值，而是**只用它做排序**，不做截断决策——截断由 top-K 决定。对 logits 我做了 sigmoid 映射到 [0,1]，但明确知道它不一定是概率。如果真想用分数做"低于 X 就不回答"这种决策，我们系统里用的是**置信度启发式**（基于来源数量和平均分数组合出来的 confidence），而不是直接拿模型分数当概率用。

**【面试官可能追问】top_k * 3 的候选扩展是怎么做的？会不会把噪声也一起扩进来？**

应答策略：我们检索时先拉 top-20 候选，再用 cross-encoder 精排回 top-5，这就是标准的"扩召回收紧精度"思路。要回答"会不会扩进噪声"——会，但这是**有意为之**：cross-encoder 的价值恰恰在于它能纠正 bi-encoder 的排序错误，如果你只给它 5 个候选，等于假设 bi-encoder 前 5 已经是对的，那就失去精排意义了。所以候选数要大于最终数，形成一个"recall 池"。如果你担心噪声，可以在扩池时做一层**类别/知识点过滤**，我们后续加过基于图谱节点的过滤。补充一句：扩池的代价是延迟线性上涨，所以 top-20 这个数是延迟和质量之间压出来的平衡点。
