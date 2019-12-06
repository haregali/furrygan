import torch.utils.data as data
import os
from PIL import Image
import random
import torchvision.transforms as transforms
import time
import torch
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.npy',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    print(dir)
    for root, _, fnames in sorted(os.walk(dir))[:10]:
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class DataLoader(data.Dataset):
    def __init__(self, batchSize, isTrain=True, data_aug=True):

        self.batchSize = batchSize
        self.data_aug = data_aug
        
        self.phase = 'train' if isTrain else 'test'
        self.root_dir = "//home/prateek_jain3130/furrygan/dataset/"

        dir_A = self.phase + '_label'
        self.dir_A = os.path.join(self.root_dir, dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        dir_B = self.phase + '_img'
        self.dir_B = os.path.join(self.root_dir, dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)
        


    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = np.load(A_path)
        
        A = Image.fromarray(A.astype('uint8'))        

        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        
        
#         scale_size = float(self.opt.longSize/max_size)
#         new_w = int(scale_size * w)
#         new_h = int(scale_size * h)
        A = A.resize((256,256),Image.NEAREST)
        # if self.opt.isTrain or self.opt.random_embed==False:
#         B_path = self.B_paths[index]
#         B = Image.open(B_path).convert('RGB')
        B = B.resize((256,256),Image.BICUBIC)

        C_tensor = 0

        A_tensor = transforms.functional.to_tensor(A) * 255.0
        B_tensor = transforms.functional.to_tensor(B)
        real_B_tensor = B_tensor.clone()
        mask_bg = (A_tensor==0).type(torch.FloatTensor)
        B_tensor = torch.clamp(B_tensor + mask_bg*torch.ones(A_tensor.size()),0,1)
        B = transforms.functional.to_pil_image(B_tensor)
        
        C_tensor = 0
        
        if self.data_aug:
            rotate,scale,shear = random.random()-0.5, random.random()-0.5, random.random()-0.5
            rotate,scale,shear = 0,0,0

            B = transforms.functional.affine(B, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.BICUBIC)
            A = transforms.functional.affine(A, 20*rotate,[0,0],1+0.2*scale,10*shear,resample=Image.NEAREST)

            C_tensor = transforms.functional.to_tensor(B)
            C_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C_tensor)
            
        B_tensor = transforms.functional.to_tensor(B)
        B_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B_tensor)
        real_B_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(real_B_tensor)
        
        A_tensor = transforms.functional.to_tensor(A) * 255.0
        
        mask_tensor = torch.zeros(6)
        
        try:
            mask_left_eye_r = torch.nonzero(A_tensor==4)
            this_top = int(torch.min(mask_left_eye_r,0)[0][1])
            this_left = int(torch.min(mask_left_eye_r,0)[0][2])
            this_bottom = int(torch.max(mask_left_eye_r,0)[0][1])
            this_right = int(torch.max(mask_left_eye_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor[0] = y_mean
            mask_tensor[1] = x_mean
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("left eye problem ------------------")
            print(A_path)
            mask_tensor[0] = 116
            mask_tensor[1] = 96
            # mask_list.append(116)
            # mask_list.append(96)

        try:
            mask_right_eye_r = torch.nonzero(A_tensor==5)
            this_top = int(torch.min(mask_right_eye_r,0)[0][1])
            this_left = int(torch.min(mask_right_eye_r,0)[0][2])
            this_bottom = int(torch.max(mask_right_eye_r,0)[0][1])
            this_right = int(torch.max(mask_right_eye_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor[2] = y_mean
            mask_tensor[3] = x_mean
            # mask_list.append(x_mean)
            # mask_list.append(y_mean)
        except:
            print("right eye problem --------------")
            print(A_path)
            mask_tensor[2] = 116
            mask_tensor[3] = 160

            # mask_list.append(116)
            # mask_list.append(160)

        try:
            mask_mouth_r = torch.nonzero((A_tensor==7)+(A_tensor==8)+(A_tensor==9))
            this_top = int(torch.min(mask_mouth_r,0)[0][1])
            this_left = int(torch.min(mask_mouth_r,0)[0][2])
            this_bottom = int(torch.max(mask_mouth_r,0)[0][1])
            this_right = int(torch.max(mask_mouth_r,0)[0][2])
            x_mean = int((this_left+this_right)/2)
            y_mean = int((this_top+this_bottom)/2)
            mask_tensor[4] = y_mean
            mask_tensor[5] = x_mean

        except:
            print("mouth problem --------------")
            print(A_path)
            mask_tensor[4] = 184
            mask_tensor[5] = 128
            
        inst_tensor = feat_tensor = 0
        append_A_tensor = self.append_region(A,A_tensor,mask_tensor)
#         append_A_tensor = append_A_tensor.reshape((1, 1, 512, 512))
#         print(append_A_tensor.shape)
        
        input_dict = {'label': append_A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'bg_image': real_B_tensor, 'ori_label': A_tensor,
                      'feat': feat_tensor, 'path': A_path, 'image_affine': C_tensor, 'mask': mask_tensor}
        
        return input_dict
        # return 1
    
    def append_region(self,label,face_label,mask_tensor):
        w,h = label.size    
        new_w = int(1.1 * w)
        new_h = int(1.1 * h)
        label_scale = label.resize((new_w,new_h),Image.NEAREST)

        label_scale_tensor = transforms.functional.to_tensor(label_scale) * 255.0
        mask_tensor_scale = torch.zeros(6)        
        mask_tensor_diff = torch.zeros(6)
        for index in range(6):
            mask_tensor_scale[index] = int(1.1*mask_tensor[index])
            mask_tensor_diff[index] = int(mask_tensor_scale[index]-mask_tensor[index])

        # left_eye = label_scale.crop((mask_tensor_diff[0],mask_tensor_diff[1],mask_tensor_diff[0]+w,mask_tensor_diff[1]+h))
        # right_eye = label_scale.crop((mask_tensor_diff[2],mask_tensor_diff[3],mask_tensor_diff[2]+w,mask_tensor_diff[3]+h))
        # mouth = label_scale.crop((mask_tensor_diff[4],mask_tensor_diff[5],mask_tensor_diff[4]+w,mask_tensor_diff[5]+h))

        left_eye_mask_whole = label_scale_tensor[:,int(mask_tensor_diff[0]):int(mask_tensor_diff[0])+h,int(mask_tensor_diff[1]):int(mask_tensor_diff[1])+w]
        right_eye_mask_whole = label_scale_tensor[:,int(mask_tensor_diff[2]):int(mask_tensor_diff[2])+h,int(mask_tensor_diff[3]):int(mask_tensor_diff[3])+w]
        mouth_mask_whole = label_scale_tensor[:,int(mask_tensor_diff[4]):int(mask_tensor_diff[4])+h,int(mask_tensor_diff[5]):int(mask_tensor_diff[5])+w]

        left_eye_mask = (left_eye_mask_whole==4).type(torch.FloatTensor)
        right_eye_mask = (right_eye_mask_whole==5).type(torch.FloatTensor)
        mouth_mask = ((mouth_mask_whole==7)+(mouth_mask_whole==8)+(mouth_mask_whole==9)).type(torch.FloatTensor)

        face_label = left_eye_mask*left_eye_mask_whole + (1-left_eye_mask)*face_label
        face_label = right_eye_mask*right_eye_mask_whole + (1-right_eye_mask)*face_label
        face_label = mouth_mask*mouth_mask_whole + (1-mouth_mask)*face_label

        return face_label
    
    def __len__(self):
        return len(self.A_paths) // self.batchSize * self.batchSize


# dataset = CreateDataset()
