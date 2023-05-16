在 torch 中如何集成 tensorboard 进行可视化、调参等？


```python
from torch.utils.tensorboard import SummaryWriter



self.writer     = SummaryWriter(self.log_dir)
self.writer.add_graph(model, dummy_input)
self.writer.add_scalar('val_loss', val_loss, epoch)
```