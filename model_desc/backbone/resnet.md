


## 具体方法

形状变化
1、第一个kernel是 7*7， 但是现在流行的实现都改成了 3*3

形状变化 
stage1, stage2, stage3, stage4, stage5
224，    56，     28，     14，    7
1,       4,       8,       16,     32


resnet 会使用 BN 层.
