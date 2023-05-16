torch 的 dataloader 部分有哪些技巧？和需要注意的点？


# 参考
[Transformers多机多卡的炼丹实践](https://blog.csdn.net/nghuyong/article/details/127959411) 学习 Huggingface, 内置trainer处理分布式训练问题。
[详解torch中的collate_fn参数](https://zhuanlan.zhihu.com/p/493400057) 


训练大模型一定是基于大数据，可能非常大（例如上百GB），所以不能采用map-style的dataset作为训练集的dataset，因为无法直接load到内存中，所以需要采用IterableDataset。同时为了训练的数据较快，需要采用多进程的数据加载，即num_worker>0。

```python
class CustomIterableDataset(IterableDataset):
    
    def __init__(self, data_file):
        self.data_file = data_file
    
    def __iter__(self):
        while True:
            with open(self.data_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    print(f"系统进程号{os.getpid()}, 加载的数据{line.strip()}")
                    yield line.strip()
```


```python
import torch.utils.data as data

class MyDataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
```