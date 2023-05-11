# Cascade R-CNN
[Cascade R-CNN 详细解读](https://zhuanlan.zhihu.com/p/42553957)

## 
解决 mismatch 问题
- 在training阶段，由于我们知道gt，所以可以很自然的把与gt的iou大于threshold（0.5）的Proposals作为正样本，这些正样本参与之后的bbox回归学习。
- 在inference阶段，由于我们不知道gt，所以只能把所有的proposal都当做正样本，让后面的bbox回归器回归坐标。
training阶段和inference阶段，bbox回归器的输入分布是不一样的，training阶段的输入proposals质量更高(被采样过，IoU>threshold)，inference阶段的输入proposals质量相对较差（没有被采样过，可能包括很多IoU<threshold的），这就是论文中提到mismatch问题，这个问题是固有存在的，通常threshold取0.5时，mismatch问题还不会很严重。

## 具体方案
像是给下一个stage 设置了 IoU 阈值，一级比一级高。
