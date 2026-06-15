# Role & Task
你是一个异构计算（NPU加速）与大模型分布式推理领域的资深专家，精通 vLLM 架构、昇腾（Ascend）底层算子调度、多流（Multi-stream）管理以及高级推理优化技术（如 PD分离、MTP、KV Cache 池化）。

现在我们需要定位一个在特定代码变更（Commit 演进）后引入的 **Decode 节点流挂死（Stream Hang）** 现象。请根据我提供的上下文、复现路径和基线对比，帮我精准定位可能出问题的代码模块、根因方向，并给出排查建议。

---

# 1. 环境与复现上下文 (Context & Environment)
* **硬件/平台环境**：昇腾 A3 场景（Ascend A3 硬件平台环境）
* **模型配置**：DeepSeek V3.1
* **部署模式**：Prefill / Decode 分离（PD 分离）场景
* **特性开关组合（核心变量）**：
    * MTP = 2 (Multi-token Prediction)
    * FULL_DECODE_ONLY = 打开
    * lmhead = 打开
    * recompute (重计算) = 打开
    * 多流 (Multi-stream) = 打开
    * KV Cache 池化 (Pooling) = 打开
* **测试任务**：GSM8K 精度推理

---

# 2. 故障现象与关键定位线索 (The Issue & Clues)
* **故障现象**：在上述配置下运行 GSM8K 时，**D（Decode）节点发生流挂死（Stream Hang）**。
* **关键消元法线索**：**关闭 AIV（AI Vector 算子核心/矢量核）后，推理恢复正常，不再挂死。**

---

# 3. 版本对比基线 (Git Baseline)
该问题属于**代码演进过程中引入的 Regression**。以下是正常版本（基线）与当前问题版本的对比：

### 【正常版本（OK Base）】
在这个配套版本下，上述场景完全正常，无挂死：
* **vLLM**: 分支 `v0.20.2`，Commit ID: `bc150f50299199599673614f80d12a196f377655`
* **vllm-ascend**: `main` 分支，Commit ID: `ff4807eafa60cf588c7e6c3d4baa767a104c27af`

### 【问题版本（NG Target）】
* **当前测试环境**：使用了比上述基线**更新**的 vLLM 和 vllm-ascend 代码（即基线之后的某次代码修改引入了该 Bug）。

---

# 4. 需要你分析并输出的方向 (Output Requirements)
请基于上述高度复杂的组合场景（MTP2 + 多流 + 池化 + recompute + AIV），结合基线到最新代码的演进可能性，从以下几个维度进行深度剖析：

1. **AIV（矢量核）相关的硬件/调度冲突**：
   为什么“关闭 AIV 正常，开启 AIV 挂死”？在多流和池化打开的情况下，AIV 开启通常涉及哪些异步流水、同步锁（Event/Stream Sync）或者内存/显存（HBM）的竞争？
2. **多流（Multi-stream）与同步机制冲突**：
   在 `FULL_DECODE_ONLY` 叠加 `MTP=2`、`recompute` 和`池化`时，D 节点的流管理非常复杂。后续的修改可能在哪个模块（例如：Stream 之间缺少 `wait_stream`，或者多流交织下 AIV 算子的非阻塞下发导致了死锁）引入了 Hang？
3. **内存池化（Pooling）与重计算（Recompute）的地址冲突**：
   池化和重计算会频繁复用/释放内存。后续修改是否可能引入了非法地址访问，导致 AIV 内部硬件报错进而卡死整个 Stream？
4. **排查策略与关键 Diff 走查点**：
   建议我们优先去走查 vLLM 和 vllm-ascend 哪些模块（如：runtime 调度、ascend 算子桥接层、C++ 侧的 stream 同步逻辑）在基线 commit 之后的修改？请列出 Top 3 怀疑点。
