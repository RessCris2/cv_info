如何加载 pretrained models 的权重？



```python
from utils.utils import download_weights

#----------------------------------------------------#
#   下载预训练权重
#----------------------------------------------------#
if pretrained:
    if distributed:
        if local_rank == 0:
            download_weights(backbone)  
        dist.barrier()
    else:
        download_weights(backbone)

if not pretrained:
    weights_init(model)
if model_path != '':
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    if local_rank == 0:
        print('Load weights {}.'.format(model_path))
    
    #------------------------------------------------------#
    #   根据预训练权重的Key和模型的Key进行加载
    #------------------------------------------------------#
    model_dict      = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location = device)
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    #------------------------------------------------------#
    #   显示没有匹配上的Key
    #------------------------------------------------------#
    if local_rank == 0:
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
```
