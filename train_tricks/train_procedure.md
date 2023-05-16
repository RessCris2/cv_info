


```python
model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()


```


# 保存权值

```python
#-----------------------------------------------#
#   保存权值
#-----------------------------------------------#
if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
    torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
    print('Save best model to best_epoch_weights.pth')
    torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

```



# 保存模型

```python
torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
```