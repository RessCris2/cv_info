
参考
[PyTorch | 保存和加载模型](https://zhuanlan.zhihu.com/p/82038049)

## 保存模型
```python
torch.save(model.state_dict(), PATH)

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval() # 用于预测时使用
model.train()  # 用于继续训练时使用

```

## 如果是保存 checkpoint
是不是需要保存 optimizer， loss , epoch 之类的？
