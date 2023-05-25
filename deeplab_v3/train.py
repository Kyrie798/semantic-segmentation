import os 
import torch
import numpy as np
import datetime
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler as GradScaler

from utils.history import LossHistory, EvalCallback
from utils.dataset import deeplab_dataset_collate, DeeplabDataset
from utils.utils_fit import fit_one_epoch
from model.deeplabv3 import DeepLab
from model.deeplabv3_training import get_lr_scheduler, set_optimizer_lr

if __name__ == "__main__":
    fp16 = True # 混合精度训练

    num_classes = 21

    model_path = "model_data/deeplab_mobilenetv2.pth"
    input_shape = [512, 512]

    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8

    UnFreeze_Epoch = 30 # 总epoch
    Unfreeze_batch_size = 4

    Freeze_Train = True

    Init_lr = 7e-3
    Min_lr = Init_lr * 0.01

    save_period = 5 # 每5个epoch保存一次权值
    save_dir = "logs"

    dice_loss = False
    focal_loss = False # 防止正负样本不平衡
    cls_weights = np.ones([num_classes], np.float32) # 给不同种类赋予不同的损失权值

    train_annotation_path = "VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
    val_annotation_path = "VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))

    model = DeepLab(num_classes=num_classes, downsample_factor=16)
    
    # 导入预训练权重
    if model_path != "":
        print("Load weights {}.".format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("Successful Load Key:{}. \nSuccessful Load Key Num:{}".format(str(load_key)[:500], len(load_key)))
        print("Fail Load Key:{}. \nFail Load Key Num:{}".format(str(no_load_key)[:500], len(no_load_key)))

    # 记录loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(),"%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # 混合精度训练
    if fp16:
        scaler = GradScaler()

    model_train = model.train()
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()

    # 读取数据集路径
    with open(train_annotation_path, "r") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    # 冻住主干网络
    UnFreeze_flag = False
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs = 16
    lr_limit_max = 1e-1
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 定义optimizer
    optimizer = optim.SGD(model.parameters(), Init_lr_fit, momentum=0.9, nesterov=True, weight_decay=1e-4)

    # 获得学习率下降的公式
    lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True)
    val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate)

    # 记录eval的map曲线
    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, "VOCdevkit", log_dir, True, eval_flag=True, period=5)

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 如果当前epoch>冻住的epoch，则解冻
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 16
                lr_limit_max = 1e-1
                lr_limit_min = 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate)
                val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate)

                UnFreeze_flag = True
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader, val_dataloader, 
                        UnFreeze_Epoch, device, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir)
    loss_history.writer.close()