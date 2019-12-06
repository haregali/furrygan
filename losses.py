import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
# class MFMLoss(nn.Module):
#     def __init__(self):
#         super(MFMLoss, self).__init__()
#         self.criterion = nn.MSELoss()

#     def forward(self, x_input, y_input):
#         loss = 0
#         for i in range(len(x_input)):
#             x = x_input[i][-2]
#             y = y_input[i][-2]
#             assert x.dim() == 4 
#             assert y.dim() == 4
#             x_mean = torch.mean(x,0)
#             y_mean = torch.mean(y,0)
#             loss += self.criterion(x_mean, y_mean.detach())
#         return loss   

# class Vgg19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         vgg_pretrained_features = models.vgg19(pretrained=True).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, X, layers_num=5):
#         h_relu1 = self.slice1(X)
#         if layers_num == 1:
#             return [h_relu1]   
#         h_relu2 = self.slice2(h_relu1)     
#         if layers_num == 2:
#             return [h_relu1, h_relu2]   
#         h_relu3 = self.slice3(h_relu2)   
#         if layers_num == 3:
#             return [h_relu1, h_relu2, h_relu3]     
#         h_relu4 = self.slice4(h_relu3)        
#         if layers_num == 4:
#             return [h_relu1, h_relu2, h_relu3, h_relu4]     
#         h_relu5 = self.slice5(h_relu4)                
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out

# class VGGLoss(nn.Module):
#     def __init__(self, weights = None):
#         super(VGGLoss, self).__init__()       
#         if weights != None: 
#             self.weights = weights
#         else:
#             self.weights = [1.0/4, 1.0/4, 1.0/4, 1.0/8, 1.0/8]        
#         self.vgg = Vgg19()
#         self.criterion = nn.L1Loss()

#     def forward(self, x, y, face_mask, mask_weights):              
#         assert face_mask.size()[1] == len(mask_weights)  # suppose to be 5
#         x_vgg, y_vgg = self.vgg(x,layers_num=len(self.weights)), self.vgg(y,layers_num=len(self.weights))
#         mask = []
#         mask.append(face_mask.detach())
        
#         downsample = nn.MaxPool2d(2)
#         for i in range(len(x_vgg)):
#             mask.append(downsample(mask[i]))
#             mask[i] = mask[i].detach()
#         loss = 0
#         for i in range(len(x_vgg)):
#             for mask_index in range(len(mask_weights)):
#                 a = x_vgg[i]*mask[i][:,mask_index:mask_index+1,:,:]
#                 loss += self.weights[i] * self.criterion(x_vgg[i]*mask[i][:,mask_index:mask_index+1,:,:], (y_vgg[i]*mask[i][:,mask_index:mask_index+1,:,:]).detach()) * mask_weights[mask_index]
#         return loss    

# class GramMatrixLoss(nn.Module):
#     def __init__(self):
#         super(GramMatrixLoss, self).__init__()        
#         self.weights = [1.0,1.0,1.0]
#         self.vgg = Vgg19()
#         # self.criterion = nn.L1Loss()
#         self.criterion = nn.MSELoss()
#         # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

#     def forward(self, x, y, label):
#         # we use this label to label face
#         face_mask = (label==1).type(torch.FloatTensor)
#         mask = []
#         mask.append(face_mask)
#         x_vgg, y_vgg = self.vgg(x,layers_num=len(self.weights)), self.vgg(y,layers_num=len(self.weights))
#         downsample = nn.MaxPool2d(2)
#         for i in range(len(x_vgg)):
#             mask.append(downsample(mask[i]))
#             mask[i] = mask[i].detach()
#         loss = 0
#         for i in range(len(x_vgg)):
#             loss += self.weights[i] * self.criterion(grammatrix(x_vgg[i]*mask[i]), grammatrix(y_vgg[i]*mask[i]).detach())
#         return loss