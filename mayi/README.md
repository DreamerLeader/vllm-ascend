# mooncakeconnectorV1运行指南

## 环境要求

* 软件：
  * Python >= 3.9， < 3.12
  * CANN >= 8.2.rc1
  * PyTorch >= 2.5.1，torch-npu >= 2.5.1.post1.dev20250619
  * vLLM（与 vllm-ascend 版本相同）

vllm的版本要与vllm-ascend的main分支一致，2025/07/30，版本为

* vllm: v0.10.0
* vllm-ascend: main分支

## 运行

### 1、mooncake.json配置

```json
{
  "prefill_url": "localhost:15272",
  "decode_url": "localhost:14012",	
  "metadata_backend": "http",
  "protocol": "ascend"
}
```

"prefill\_url": 配置prefill所在ip和prot,<br>
"decode\_url": 配置decode所在ip和prot",<br>
"metadata\_backend": 选http,<br>
"protocol": 配置hccl的通信, 配置ascend使用hccl的通信方式<br>

### 2、拉起`prefill`节点

```bash
bash run_prefill.sh
```

run_prefill.sh脚本内容

```bash
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=localhost
export GLOO_SOCKET_IFNAME="xxxxxx"
export TP_SOCKET_IFNAME="xxxxxx"
export HCCL_SOCKET_IFNAME="xxxxxx"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCAKE_CONFIG_PATH="/xxxxx/mooncake.json"
export VLLM_USE_V1=1
export VLLM_BASE_PORT=9700

vllm serve "/xxxxx/DeepSeek-V2-Lite-Chat" \
  --host localhost \
  --port 8100 \
  --tensor-parallel-size 2\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000  \
  --trust-remote-code \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-address localhost \
  --data-parallel-rpc-port 9100 \
  --gpu-memory-utilization 0.8  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_rank": 0,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 2
             }
      }
  }'  \
```
`HCCL_EXEC_TIMEOUT`、`HCCL_CONNECT_TIMEOUT`、`HCCL_IF_IP`为hccl相关配置<br>
`GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME`配置为对应网卡<br>
`ASCEND_RT_VISIBLE_DEVICES`指定节点运行在哪些卡上，卡总数等于dp*tp<br>
`OMP_PROC_BIND`、`OMP_NUM_THREADS`默认配置<br>
`MOONCAKE_CONFIG_PATH`指定mooncake.json的路径<br>
`VLLM_USE_V1`配置为1<br>
`VLLM_BASE_PORT`配置vllm的基础port<br>
`/xxxxx/DeepSeek-V2-Lite-Chat`配置为需要运行的模型<br>
`--host`配置为拉起节点所在ip<br>
`--port`配置拉起的port，与步骤4中的port对应<br>
`--seed`、`--max-model-len`、`--max-num-batched-tokens`模型基础配置，按照实际场景配置<br>
`--tensor-parallel-size`：配置tp的size<br>
`--data-parallel-size`：配置dp的size<br>
`--data-parallel-address`：dp的ip，配置为节点所在ip<br>
`--data-parallel-rpc-port`：dp分组中通信的rpc port<br>
`--trust-remote-code`能够加载自己本地的模型<br>
`--enforce-eager`不开图模式<br>
`--gpu-memory-utilization`占用卡的显存比例<br>
`--kv-transfer-config`：关注kv_connector、kv_connector_module_path按脚本中的配置使用mooncakeconnect，kv_buffer_device指定运行在npu卡上，kv_role配置kv_producer为p节点，配置kv_consumer为d节点，kv_parallel_size并行配置默认为1，kv_port节点连接使用的port，对于p节点engine_id、kv_rank配置为0，d节点配置为1；kv_connector_extra_config中配置p、d节点的分布式并行策略，按照上面--tensor-parallel-size与--data-parallel-size配置<br>

### 3、拉起`decode`节点

```bash
bash run_decode.sh
```

run_decode.sh脚本内容

```bash
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=localhost
export GLOO_SOCKET_IFNAME="xxxxxx"
export TP_SOCKET_IFNAME="xxxxxx"
export HCCL_SOCKET_IFNAME="xxxxxx"
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export MOONCAKE_CONFIG_PATH="/xxxxx/mooncake.json"
export VLLM_USE_V1=1
export VLLM_BASE_PORT=9700

vllm serve "/xxxxx/DeepSeek-V2-Lite-Chat" \
  --host localhost \
  --port 8200 \
  --tensor-parallel-size 2\
  --seed 1024 \
  --max-model-len 2000  \
  --max-num-batched-tokens 2000  \
  --trust-remote-code \
  --enforce-eager \
  --data-parallel-size 2 \
  --data-parallel-address localhost \
  --data-parallel-rpc-port 9100 \
  --gpu-memory-utilization 0.8  \
  --kv-transfer-config  \
  '{"kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_parallel_size": 1,
  "kv_port": "20002",
  "engine_id": "1",
  "kv_rank": 1,
  "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 2,
                    "tp_size": 2
             },
             "decode": {
                    "dp_size": 2,
                    "tp_size": 2
             }
      }
  }'  \
```

### 4、启动proxy\_server

```bash
cd /vllm-ascend/examples/disaggregate_prefill_v1/
bash run_proxy.sh
```

run_proxy.sh脚本内容

```bash
python toy_proxy_server.py \
    --host  localhost\
    --prefiller-hosts localhost  host1 \
    --prefiller-ports 8100 8101 \
    --decoder-hosts localhost host2 \
    --decoder-ports 8200 8201 \
```

`--host`：为主节点，步骤5中的curl命令下发中的localhost与该host保持一致，拉起服务代理端口默认8000<br>
`--prefiller-hosts`：配置为所有p节点的ip，对于xpyd的场景，依次将ip写在该配置后面，ip与ip之间空一格<br>
`--prefiller-ports`：配置为所有p节点的port，也是步骤3中vllm拉起服务--port的配置， 依次将port写在该配置后面，port与port之间空一格，且顺序要保证和--prefiller-hosts的ip一一对应<br>
`--decoder-hosts`：配置为所有d节点的ip，对于xpyd的场景，依次将ip写在该配置后面，ip与ip之间空一格<br>
`--decoder-ports`：配置为所有d节点的port，也是步骤4中vllm拉起服务--port的配置， 依次将port写在该配置后面，port与port之间空一格，且顺序要保证和--decoder-hosts的ip一一对应<br>

### 5、推理任务下发

推理中的ip配置成自己的
其中model变量配置成自己模型的路径，同时保证和shell脚本里面的一致

```bash
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
"model": "model_path",
"prompt": "Given the accelerating impacts of climate change—including rising sea levels, increasing frequency of extreme weather events, loss of biodiversity, and adverse effects on agriculture and human health—there is an urgent need for a robust, globally coordinated response. However, international efforts are complicated by a range of factors: economic disparities between high-income and low-income countries, differing levels of industrialization, varying access to clean energy technologies, and divergent political systems that influence climate policy implementation. In this context, how can global agreements like the Paris Accord be redesigned or strengthened to not only encourage but effectively enforce emission reduction targets? Furthermore, what mechanisms can be introduced to promote fair and transparent technology transfer, provide adequate financial support for climate adaptation in vulnerable regions, and hold nations accountable without exacerbating existing geopolitical tensions or disproportionately burdening those with historically lower emissions?",
"max_tokens": 256
}'
```








