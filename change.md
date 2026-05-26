最小化我建议只改 `model_runner_v1.py`，先不要动 `pool_worker.py/register_buffer`。

核心改法：把 DSV4 `use_compress` 场景的 `positions` 恢复成正常分支的 CPU staging 路径。

改 3 个地方：

1. [model_runner_v1.py:378](/Users/pengbozhao/workspace/vllm-ascend/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:378)

把 PCP/DCP 下这个：

```python
self.positions = torch.zeros(
    max_buffer_num_tokens, dtype=torch.int64, device=self.device)
```

改回：

```python
self.positions = self._make_buffer(max_buffer_num_tokens, dtype=torch.int64)
```

2. [model_runner_v1.py:2808](/Users/pengbozhao/workspace/vllm-ascend/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2808)

把：

```python
positions=self.positions,
positions_cpu=self.positions.cpu() if self.use_compress else None,
```

改成兼容写法：

```python
positions=self.positions.gpu if hasattr(self.positions, "gpu") else self.positions,
positions_cpu=self.positions.cpu if hasattr(self.positions, "cpu") else (
    self._positions_cpu_buf[:num_tokens] if self.use_compress else None
),
```

关键是：`positions_cpu` 不能再用 `self.positions.cpu()`，这个是实时 D2H sync。

3. [model_runner_v1.py:3133](/Users/pengbozhao/workspace/vllm-ascend/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3133)

graph capture dummy input 这里：

```python
if self.use_compress:
    self.positions.fill_(127)
```

改成：

```python
if self.use_compress:
    if hasattr(self.positions, "np"):
        self.positions.np.fill(127)
        self.positions.copy_to_gpu()
    else:
        self.positions.fill_(127)
```

结论很直接：不要通过跳过 `register_buffer` 修；`register_buffer` 只是把问题暴露出来。最小修复应该让 DSV4 compressed attention 回到正常分支那种 `CpuGpuBuffer positions + CPU metadata staging`，避免 graph capture / prefill metadata 里临时从 NPU tensor 做 Python 控制流或 D2H 拷贝。
