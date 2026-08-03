# 亮点二：标准化 Agent 执行框架（BaseAgent + Harness）

> **核心数字**：样板代码 -60%｜失败率 8%→0.3%｜排查从"翻 6 份代码"到"看 6 行日志"

## 1. 背景阐述

接着上一个亮点往下说。当我们把 6 个 Agent 都跑起来之后，很快就发现一个新问题：**每个 Agent 的代码长得一模一样**。你去看 Designer、Coder、Assessment 的源码，会发现每个里面都有一段差不多的东西——try/except 包住 LLM 调用、出错重试、记耗时、打日志。Phase 3 的时候每个 Agent 大约有 50 行这种样板代码，6 个 Agent 就是 300 行，而且每个的写法还不完全一样。

## 2. 问题剖析

虽然"每个 Agent 能跑"看起来很美好，但**一旦要改"重试策略"或者"日志格式"，就要同时改 6 个文件**，而且很容易改漏。

更麻烦的是**行为不一致**：有的 Agent 遇到 LLM 超时会重试 3 次，有的重试 1 次，有的干脆不重试直接抛异常；有的 Agent 异常被吞掉了，日志里什么都查不到。你要知道，多 Agent 系统最大的排查难点就是"**到底哪一步失败了、为什么失败**"——没有统一的执行报告，出问题你只能一个个 Agent 去看代码。如果放任不管，系统维护成本会指数级上升，后面加第 7 个、第 8 个 Agent 的时候，等于每加一个都要把错误处理的重活再干一遍。

## 3. 方案构思与技术实现

所以 Phase 7 我做了个决定：**抽一个统一的执行框架（我们内部叫 Harness），所有 Agent 都往这个基类上收敛**。

技术选型上，我纠结过两个方向：一个是"**深继承**"——把所有能力都塞进 `BaseAgent` 基类，子类只要继承就有；另一个是"**基类 + 可选 Mixin**"。我选了后者。原因很简单：**不是所有 Agent 都需要工具和记忆**——Coder 不需要调工具，Guardian 根本不需要 LLM。如果全塞基类里，每个子类都背着用不到的能力，违反接口隔离原则。所以 `BaseAgent` 只保证最核心的三阶段生命周期，工具和记忆用 `ToolInjectionMixin` / `MemoryAwareMixin` 按需混入。

核心代码是 `execute()` 这个统一入口，你看它把整个生命周期管起来了：

```python
class BaseAgent(ABC):
    async def execute(self, input: AgentInput) -> ExecutionReport:
        report = ExecutionReport(agent_name=self._agent_name)
        timer = Timer(); timer.start()

        try:
            # ① before_run：记忆加载、上下文增强（带重试）
            input = await RetryHandler.with_retry(
                self.before_run, input,
                max_retries=self._config.before_retries,
            )
            report.memory_context_loaded = input.memory_context is not None

            # ② run：核心业务逻辑（子类实现，带重试）
            retry_count = [0]
            output = await RetryHandler.with_retry(
                self.run, input,
                max_retries=self._config.max_retries,
                on_retry=lambda a, e: retry_count.__setitem__(0, a + 1),
            )
            report.retries = retry_count[0]

            # ③ after_run：落库、审计
            output = await self.after_run(output)
            report.success = True
            report.output = output

        except AgentError as exc:
            report.success = False
            report.error = str(exc)
            report.error_type = type(exc).__name__
        except Exception as exc:
            report.success = False
            report.error = str(exc)
            report.error_type = type(exc).__name__

        report.duration_ms = timer.stop()
        log_execution_report(report)   # 结构化 JSON 日志
        return report
```

再配合重试处理器，指数退避，可配置哪些异常可重试：

```python
class RetryHandler:
    DEFAULT_RETRYABLE = (TimeoutError, httpx.TimeoutException,
                         httpx.RequestError, ConnectionError)

    @staticmethod
    async def with_retry(func, *args, max_retries=3, base_delay=1.0, ...):
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except retryable as exc:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                    await asyncio.sleep(delay)
        raise last_exc
```

**为什么重试只针对网络类异常、不针对内容类异常？** 因为网络类异常（超时、连接重置）重试大概率能成功，而内容类异常（模型输出 schema 不合法）重试同一遍 prompt 大概率还是错的——那不是重试能解决的，得改 prompt 或降级。这是我在设计重试策略时最关键的判断。

**兜底细节**：记忆加载用的是 `before_run`，但它挂了**不能影响主流程**——所以里面用 try/except 包住，失败了打 warning 继续往下走。这个原则很重要：**外围能力（记忆、工具）不能成为核心链路的单点故障**。

## 4. 优化效果与压测数据

这个改造不改变 LLM 本身的耗时，但它带来了几个可以量化的收益：

- **样板代码量**：6 个 Agent 迁移后，每个 Agent 平均减少约 50 行重复代码，**总代码量精简约 60%**
- **重试行为统一**：之前有的 Agent 不重试、有的重试 3 次，迁移后全部收敛到统一策略。我们压测了"模拟 LLM 接口随机 10% 概率超时"的场景：
  - 改造前：约 **8% 的 Agent 调用直接失败**（因为有的不重试），成功请求的平均重试时间 **约 3.1s**（因为退避不统一）
  - 改造后：失败率降到 **0.3%**（3 次重试后仍失败的极少数），每次重试的平均额外耗时约 **2.1s**（1s+2s 退避，比之前更规范）
- **可观测性**：每个 Agent 现在都会产出一份 `ExecutionReport`，包含 success、duration_ms、retries、token 消耗，结构化日志一行打出来。**排查一次链路问题从"翻 6 份代码"降到"看 6 行日志"**

## 5. 复盘总结

核心价值就一句：**多 Agent 系统的规模不是靠加 Agent，是靠把每个 Agent 的"横切面"（重试、埋点、日志、记忆、工具）抽成统一框架**。踩过的坑是：一开始我试图让所有 Agent 走同一个 `run` 签名，结果发现 Mentor 要流式输出、Coder 要执行沙箱，形态完全不一样。最后我把"标准化的边界"定义为**执行报告和执行生命周期**，而不是输出类型——每个 Agent 的输出仍然是自己的领域模型，框架不干预。

## 6. 常见面试题与追问点

**Q1：重试是幂等的吗？如果 LLM 已经生成了内容，但响应在网络上丢了，重试会不会生成重复内容、重复入库？**

A：会，这是一个真实的风险。我们的解法分两层：第一层，Agent 的产物先存在共享的 `state["generated_resources"]` 里，**重跑之前会先清空**（我们的 `retry_prep_node` 就是干这个的）；第二层，真正入库前有**语义去重**——新资源先做向量召回，和已有知识点相似度超过阈值的直接合并，不新建节点。所以即使某个节点被重跑了，最多是浪费一点 token，不会产生脏数据。如果面试官追问"有没有用 request id 做去重"，可以答：LLM 调用是幂等的场景不多，我们更依赖"写入前幂等"，因为 LLM 本身做不到请求幂等。

**Q2：ExecutionReport 里的耗时、token 是怎么采集的？对性能有影响吗？**

A：耗时是 `time.perf_counter()` 打点，token 是从 LLM 适配器返回值里读的，都是内存操作，开销在微秒级，对秒级的 LLM 调用来说完全可以忽略。采集本身是同步的，只有落日志是 I/O，我们用的标准 logging，不会阻塞主流程。

**【面试官可能追问】Mixin 和深继承的区别？为什么工具、记忆用 Mixin 而不是全放基类？**

应答策略：核心是**接口隔离**。如果全部放基类，那么"不需要记忆的 Guardian"也得背上 `_memory_ops`、`_tool_registry` 这些字段和加载逻辑，而且 BaseAgent 的构造函数会越来越重。用 Mixin 后，Agent 可以按需声明"我需要工具"（继承 `ToolInjectionMixin`）或"我需要记忆"（继承 `MemoryAwareMixin`），基类保持稳定。另外这也是为了**可测试性**——我只测工具解析逻辑时，不需要构造一个完整的 BaseAgent 实例。如果面试官再问"那 Python 多继承的菱形问题怎么办"，可以答：Mixins 都是无状态的辅助方法集，不共享状态、不依赖 super() 链，所以不存在菱形继承的方法冲突问题。
