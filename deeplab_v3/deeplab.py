import copy
import cv2
import torch
import numpy as np
import colorsys
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from utils.utils import cvtColor, resize_image, preprocess_input
from model.deeplabv3 import DeepLab

class DeeplabV3(object):
    def __init__(self):
        self.model_path = "logs/best_epoch_weights.pth"
        self.num_classes = 21
        self.input_shape = [512, 512]

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # mix_type = 0的时候代表原图与生成的图进行混合
        # mix_type = 1的时候代表仅保留生成的图
        # mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        self.mix_type = 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepLab(num_classes=self.num_classes, downsample_factor=16)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = self.model.eval()
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def detect_image(self, image):
        image = cvtColor(image)

        # 对输入图像进行备份
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 给图像增加灰条，实现不失真的resize
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        
        # 增加batch维度
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.device)
            # 传入网络进行预测
            pr = self.model(images)[0]
            # 取出每一个像素点的种类
            pr = F.softmax(pr.permute(1,2,0), dim = -1).cpu().numpy()
            # 将灰条部分截取掉
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))
        
        return image
