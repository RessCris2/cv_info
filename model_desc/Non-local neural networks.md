# Non-local neural networks
参考资料
https://zhuanlan.zhihu.com/p/33345791 

## 提出背景
local 的含义是针对 感受野，以卷积操作为例，感受野时卷积核大小。
non-local, 比如全连接。但是参数量太大。
为什么还需要non-local: 卷积层的叠加可以增大感受野。有些任务可能需要原图上更多的信息，比如attention。如果能够在某些层引入全局的信息，就能很好地解决local操作无法看清全局的情况。


## 具体方法
![non local block](https://github.com/RessCris2/cv_info/blob/main/imgs/non_local_block_1.png)

## 为什么可以解决这个问题？