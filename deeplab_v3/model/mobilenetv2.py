import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, input, output, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = round(input * expand_ratio)
        self.use_res_connect = self.stride == 1 and input == output
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                                      nn.BatchNorm2d(hidden_dim),
                                      nn.ReLU6(True),
                                      nn.Conv2d(hidden_dim, output, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(output))
        else:
            self.conv = nn.Sequential(nn.Conv2d(input, hidden_dim, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(hidden_dim),
                                      nn.ReLU6(True),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                                      nn.BatchNorm2d(hidden_dim),
                                      nn.ReLU6(True),
                                      nn.Conv2d(hidden_dim, output, 1, 1, 0, bias=False),
                                      nn.BatchNorm2d(output))
            
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNet_V2(nn.Module):
    def __init__(self, n_class=1000, width_multi=1.):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], 
                                        [6, 24, 2, 2], 
                                        [6, 32, 3, 2], 
                                        [6, 64, 4, 2], 
                                        [6, 96, 3, 1], 
                                        [6, 160, 3, 2], 
                                        [6, 320, 1, 1]]
        input_channel = int(input_channel * width_multi)
        self.last_channel = int(last_channel * width_multi) if width_multi > 1.0 else last_channel
        self.features = [nn.Sequential(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                                       nn.BatchNorm2d(input_channel),
                                       nn.ReLU6(True))]
        
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_multi)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features.append(nn.Sequential(nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(self.last_channel),
                                           nn.ReLU6(True)))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.last_channel, n_class))
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x