import os
import torch
from tqdm import tqdm

from torch.cuda.amp import autocast
from utils.utils import get_lr
from model.deeplabv3_training import CE_Loss, Dice_loss, Focal_Loss
from utils.utils_metrics import f_score

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader, 
                  val_dataloader, Epoch, device, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0
    
    # 训练
    pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{Epoch}",postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, (imgs, pngs, labels) in enumerate(train_dataloader):
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()
            
        pbar.set_postfix(**{"total_loss": total_loss / (iteration + 1), 
                            "f_score": total_f_score / (iteration + 1),
                            "lr": get_lr(optimizer)})
        pbar.update(1)

    pbar.close()

    # 验证
    pbar = tqdm(total=epoch_step_val, desc=f"Epoch {epoch + 1}/{Epoch}",postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, (imgs, pngs, labels) in enumerate(val_dataloader):
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            _f_score = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
            pbar.set_postfix(**{"val_loss": val_loss / (iteration + 1),
                                "f_score": val_f_score / (iteration + 1),
                                "lr": get_lr(optimizer)})
            pbar.update(1)
    pbar.close()
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1, model_train)
    print("Epoch:"+ str(epoch + 1) + "/" + str(Epoch))
    print("Total Loss: %.3f || Val Loss: %.3f " % (total_loss / epoch_step, val_loss / epoch_step_val))
    
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print("Save best model to best_epoch_weights.pth")
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))