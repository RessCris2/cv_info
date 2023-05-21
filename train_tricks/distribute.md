

多卡时如何开启训练？
```python
if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
```



 多卡同步Bn
 ngpus_per_node > 1
 print("Sync_bn is not support in one gpu or not distributed.")


 torch.nn.parallel.DistributedDataParallel
 torch.nn.DataParallel


单机多卡
多机多卡： 数据并行、模型并行


# 基本概念
node_rank: 节点的编号
rank: 全局进程的编号
local_rank: 单个节点上进程的编号
word_size: 全局总进程的数量
master ip: master进程的ip地址
master port: master进程的端口