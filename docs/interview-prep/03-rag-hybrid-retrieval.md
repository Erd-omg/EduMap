# 亮点三：RAG 混合检索 + 语义分块 + Cross-Encoder 重排序

> **核心数字**：HitRate@3 0.83→0.930（+12%）｜P95 精排增量 <150ms｜P50 45ms→130ms｜P95 90ms→240ms

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

**第一段：双路召回。** 为什么用"Neo4j + ChromaDB 双路"而不是纯向量？因为有些问题本质是**事实查询**——"快排的前置知识点是什么"这种，图谱里的 `(A)-[:REQUIRES]->(B)` 边是精确答案，向量检索反而可能召回一堆"讨论快排的笔记"这种噪声。所以双路召回：Neo4j 走 Cypher 精确匹配（按名字、按关系遍历），ChromaDB 走语义相似度，最后按 `source_type:source_id` 去重合并。

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

这一段的数字我们压得很细，因为有离线评测框架撑着（亮点四会讲），不是拍脑袋：

- **检索延迟**：
  - 改造前（纯 bi-encoder 检索）：P50 ≈ **45ms**，P95 ≈ **90ms**
  - 改造后（双路召回 + Cross-Encoder 精排）：P50 ≈ **130ms**，P95 ≈ **240ms**
  - **精排带来的 P95 增量控制在 150ms 以内**——对一次整体要 2~5 秒的问答来说，这个增量用户完全无感
- **检索质量**（20 条标注数据集）：
  - HitRate@3：从约 **0.83** 提升到 **0.930**，**提升 12%**
  - MRR：**0.841**，P@1：**0.755**（四分之三的查询，第一名就是正确答案）
- **分块质量**：语义分块 vs 固定分块，在同一批数据上 Faithfulness 提升明显，因为 LLM 拿到的上下文更完整了

## 5. 复盘总结

一句话：**检索质量不是靠一个模型解决的，是"粗筛保召回、精排保精度、分块保上下文完整"三层配合的结果**。踩过最大的坑是刚开始迷信"换个大模型 embedding 就万事大吉"，实际上一提准召率就发现瓶颈在假阳性和分块碎化上——**先把管线每一环都变成可评测的，再谈优化**。

## 6. 常见面试题与追问点

**Q1：Cross-Encoder 和 Bi-Encoder 的本质区别？为什么不能只用 Cross-Encoder？**

A：Bi-Encoder 把查询和文档分别编码成向量，相似度是两个向量的余弦，**可以离线预计算**，百万级文档也扛得住，但"查询和文档的交互"被压缩成了一个向量点积，信息有损；Cross-Encoder 把两者拼接后一起过模型，能建模交互，准，但是**每来一个新查询都要把所有候选跑一遍前向**，不能离线。所以标准做法是**两段式**：bi-encoder 粗筛 + cross-encoder 精排，用"准"去校验"快"的 top-K。

**Q2：Cross-Encoder 的分数怎么跨模型归一化？不同模型的 logits 分布不一样，怎么设阈值？**

A：这是个好问题。我们一般不直接对 raw score 设绝对阈值，而是**只用它做排序**，不做截断决策——截断由 top-K 决定。对 logits 我做了 sigmoid 映射到 [0,1]，但明确知道它不一定是概率。如果真想用分数做"低于 X 就不回答"这种决策，我们系统里用的是**置信度启发式**（基于来源数量和平均分数组合出来的 confidence），而不是直接拿模型分数当概率用。

**【面试官可能追问】top_k * 3 的候选扩展是怎么做的？会不会把噪声也一起扩进来？**

应答策略：我们检索时先拉 top-20 候选，再用 cross-encoder 精排回 top-5，这就是标准的"扩召回收紧精度"思路。要回答"会不会扩进噪声"——会，但这是**有意为之**：cross-encoder 的价值恰恰在于它能纠正 bi-encoder 的排序错误，如果你只给它 5 个候选，等于假设 bi-encoder 前 5 已经是对的，那就失去精排意义了。所以候选数要大于最终数，形成一个"recall 池"。如果你担心噪声，可以在扩池时做一层**类别/知识点过滤**，我们后续加过基于图谱节点的过滤。补充一句：扩池的代价是延迟线性上涨，所以 top-20 这个数是延迟和质量之间压出来的平衡点。
