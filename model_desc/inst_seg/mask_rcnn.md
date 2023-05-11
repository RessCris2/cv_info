


## 具体方法
Mask R-CNN算法的主要步骤为：
- 首先，将输入图片送入到特征提取网络得到特征图。
- 然后对特征图的每一个像素位置设定固定个数的ROI（也可以叫Anchor），然后将ROI区域送入RPN网络进行二分类(前景和背景)以及坐标回归，以获得精炼后的ROI区域。
- 对上个步骤中获得的ROI区域执行论文提出的ROIAlign操作，即先将原图和feature map的pixel对应起来，然后将feature map和固定的feature对应起来。
- 最后对这些ROI区域进行多类别分类，候选框回归和引入FCN生成Mask，完成分割任务。
总的来说，在Faster R-CNN和FPN的加持下，Mask R-CNN开启了R-CNN结构下多任务学习的序幕。它出现的时间比其他的一些实例分割方法（例如FCIS）要晚，但是依然让proposal-based instance segmentation的方式占据了主导地位（尽管先检测后分割的逻辑不是那么地自然）



> Mask R-CNN，mask 分支的分割质量（quality）来源于检测分支的classification confidence。Mask R-CNN其实Faster R-CNN系列的延伸，其在Faster R-CNN的基础上添加一个新的分支用来预测object mask，该分支以检测分支的输出作为输入，mask的质量一定程度上依赖于检测分支。