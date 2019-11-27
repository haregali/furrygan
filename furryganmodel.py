
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import numpy as np
import functools

from losses import GANLoss
from networks import GeneratorNetwork, DiscriminatorNetwork, PNetwork, EncoderGenerator_mask_skin, EncoderGenerator_mask_eye, EncoderGenerator_mask_mouth, DecoderGenerator_mask_skin, DecoderGenerator_mask_eye, DecoderGenerator_mask_mouth, DecoderGenerator_mask_skin_image, DecoderGenerator_mask_eye_image, DecoderGenerator_mask_mouth_image


# In[2]:


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)
    
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def softmax2label(featuremap):
    _, label = torch.max(featuremap, dim=1)
    size = label.size()
    label=label.resize_(size[0],1,size[1],size[2])
    return label


# In[11]:


class FurryGAN(nn.Module):
    def __init__(self, lr=0.0002, niter_decay=0, isTrain=True):
        
        #Hyperparams
        self.lr = lr
        self.beta1 = 0.5
        self.niter_decay = niter_decay
        
        self.input_nc = 11             #number of input channels
        self.output_nc = 3             #number of output channels
        self.label_nc = 11             #number of mask channels
        self.isTrain = isTrain         #Whether to train
        self.dis_net_input_nc = self.input_nc + self.output_nc
        self.dis_n_layers = 3
        self.num_D = 2
        self.lambda_feat= 10.0
        self.z_dim = 512
        self.batch_size = opt.batch_size
        
        #Loss Function parameters - used in init_loss_funtion
        self.use_gan_feat_loss = True
        self.no_vgg_loss = True
        self.no_l2_loss = True        
        self.model_path = "checkpoints/face_parsing.pth"
        
        #Optimization Parameters
        self.use_lsgan = False
        
        self.no_ganFeat_loss= True
        
        self.gen_net = GeneratorNetwork(self.input_nc, self.output_nc)
        self.gen_net.apply(weights_init)
        
        if self.isTrain:
            use_sigmoid = True
            
        self.dis_net = DiscriminatorNetwork(self.dis_net_input_nc, self.dis_n_layers, self.num_D, use_sigmoid)
        self.dis_net.apply(weights_init)
        
        #Dont know why we need this???
        self.dis_net2 = DiscriminatorNetwork(self.dis_net_input_nc, self.dis_n_layers, self.num_D, use_sigmoid)
        self.dis_net2.apply(weights_init)
        
        
#         self.p_net = PNetwork(self.label_nc, self.output_nc)
#         self.p_net.apply(weights_init)
        
        self.p_net = PNetwork(self.model_path)
        
        
        #TODO
        longSize = 256
        n_downsample_global = 2
        embed_feature_size = longSize//2**n_downsample_global 

        self.encoder_skin_net = EncoderGenerator_mask_skin(functools.partial(nn.BatchNorm2d, affine=True))
        self.encoder_hair_net = EncoderGenerator_mask_skin(functools.partial(nn.BatchNorm2d, affine=True))
        self.encoder_left_eye_net = EncoderGenerator_mask_eye(functools.partial(nn.BatchNorm2d, affine=True))
        self.encoder_right_eye_net = EncoderGenerator_mask_eye(functools.partial(nn.BatchNorm2d, affine=True))
        self.encoder_mouth_net = EncoderGenerator_mask_mouth(functools.partial(nn.BatchNorm2d, affine=True))       
        
        self.decoder_skin_net = DecoderGenerator_mask_skin(functools.partial(nn.BatchNorm2d, affine=True))
        self.decoder_hair_net = DecoderGenerator_mask_skin(functools.partial(nn.BatchNorm2d, affine=True))
        self.decoder_left_eye_net =  DecoderGenerator_mask_eye(functools.partial(nn.BatchNorm2d, affine=True))
        self.decoder_right_eye_net = DecoderGenerator_mask_eye(functools.partial(nn.BatchNorm2d, affine=True))
        self.decoder_mouth_net =  DecoderGenerator_mask_mouth(functools.partial(nn.BatchNorm2d, affine=True)) 
        
        self.decoder_skin_image_net = DecoderGenerator_mask_skin_image(functools.partial(nn.BatchNorm2d, affine=True))
        self.decoder_hair_image_net = DecoderGenerator_mask_skin_image(functools.partial(nn.BatchNorm2d, affine=True))
        self.decoder_left_eye_image_net = DecoderGenerator_mask_eye_image(norm_layer)
        self.decoder_right_eye_image_net = DecoderGenerator_mask_eye_image(norm_layer)
        self.decoder_mouth_image_net = DecoderGenerator_mask_mouth_image(norm_layer)
        
        if self.isTrain:
            
            self.loss_filter = self.init_loss_filter(self.no_ganFeat_loss, self.no_vgg_loss, self.no_l2_loss)
            self.old_lr = self.lr
            
            self.criterionGAN = GANLoss(use_lsgan=self.use_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
#             self.criterionMFM = MFMLoss()
            
            weight_list = [0.2,1,5,5,5,5,3,8,8,8,1]
            self.criterionCrossEntropy = torch.nn.CrossEntropyLoss(weight = torch.FloatTensor(weight_list))
            
#             if self.no_vgg_loss:             
#                 self.criterionVGG = VGGLoss(weights=None)
                
#             self.criterionGM = GramMatrixLoss()
            self.loss_names = self.loss_filter('KL_embed','L2_mask_image','G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake','L2_image','ParsingLoss','G2_GAN','D2_real','D2_fake')
            
            
            params_decoder = list(self.decoder_skin_net.parameters()) + list(self.decoder_hair_net.parameters()) + list(self.decoder_left_eye_net.parameters()) + list(self.decoder_right_eye_net.parameters()) + list(self.decoder_mouth_net.parameters())
            params_image_decoder_params = list(self.decoder_skin_image_net.parameters()) + list(self.decoder_hair_image_net.parameters()) + list(self.decoder_left_eye_image_net.parameters()) + list(self.decoder_right_eye_image_net.parameters()) + list(self.decoder_mouth_image_net.parameters())
            params_encoder = list(self.encoder_skin_net.parameters()) + list(self.encoder_hair_net.parameters()) + list(self.encoder_left_eye_net.parameters()) + list(self.encoder_right_eye_net.parameters()) + list(self.encoder_mouth_net.parameters())
            
            params_together = list(self.gen_net.parameters()) + params_decoder + params_encoder + params_image_decoder
            self.optimizer_G_together = torch.optim.Adam(params_together, lr=self.lr, betas=(self.beta1, 0.999))
            
            params = list(self.dis_net.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))

            # optimizer D2
            params = list(self.dis_net2.parameters())    
            self.optimizer_D2 = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, 0.999))
                
        
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_l2_loss):
        flags = (True,True,True, use_gan_feat_loss, use_vgg_loss, True, True, use_l2_loss,True,True,True,True)
        def loss_filter(kl_loss,l2_mask_image,g_gan, g_gan_feat, g_vgg, d_real, d_fake, l2_image, loss_parsing,g2_gan,d2_real,d2_fake):
            return [l for (l,f) in zip((kl_loss,l2_mask_image,g_gan,g_gan_feat,g_vgg,d_real,d_fake,l2_image,loss_parsing,g2_gan,d2_real,d2_fake),flags) if f]
        
        return loss_filter

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.gen_net.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999))
        # if self.opt.verbose:
        #     print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_together.param_groups:
            param_group['lr'] = lr        
        # for param_group in self.optimizer_sample_net.param_groups:
        #     param_group['lr'] = lr
        # for param_group in self.optimizer_vae_net.param_groups:
        #    param_group['lr'] = lr
        # for param_group in self.optimizer_netP.param_groups:
        #    param_group['lr'] = lr            
        # for param_group in self.optimizer_vae_encoder.param_groups:
        #     param_group['lr'] = lr
        # for param_group in self.optimizer_mask_autoencoder.param_groups:
        #     param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    
    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, image_affine=None, infer=False):             
        size = label_map.size()
        oneHot_size = (size[0], self.label_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
#         if self.opt.data_type == 16:
#             input_label = input_label.half()

        # get edges from instance map
#         if not self.opt.no_instance:
#             inst_map = inst_map.data.cuda()
#             edge_map = self.get_edges(inst_map)
#             input_label = torch.cat((input_label, edge_map), dim=1) 
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # affine real images for training
        if image_affine is not None:
            image_affine = Variable(image_affine.data.cuda())

        return input_label, inst_map, real_image, feat_map, image_affine
    
    def forward(self, bg_image, label, inst, image, feat, image_affine, mask_list, ori_label, infer=False):
        
        """
        Get encoded inputs
        input_label - one hot encoded label
        inst_map - inst
        real_image - bg_image
        feat_map - feat
        real_bg_image - bg_image / image_affine
        """
        input_label, inst_map, real_image, feat_map, real_bg_image = self.encode_input(label, inst, bg_image, feat, bg_image)
        
        
        """
        Extract mask_skin and corresponding image from real_image
        mask_skin = skin + l_brow + r_brow + nose
        """
        mask_skin = ((label==1)+(label==2)+(label==3)+(label==6)).type(torch.cuda.FloatTensor)
        mask_skin_image = mask_skin * real_image
        
        
        """
        Extract mask_hair and corresponding image from real_image
        mask_hair = hair mask
        """
        mask_hair = (label==10).type(torch.cuda.FloatTensor)
        mask_hair_image = mask_hair * real_image
        
        
        
        """
        Create zero tensors for each feature image
        mask4_image - (batch_size, 3, 32, 48) left eye
        mask5_image - (batch_size, 3, 32, 48) left eye
        mask_mouth_image - (batch_size, 3, 80, 144) left eye
        """
        #Left eye and right eye
        mask4_image = torch.zeros(label.size()[0],3,32,48).cuda()
        mask5_image = torch.zeros(label.size()[0],3,32,48).cuda()
        
        #Mouth
        mask_mouth_image = torch.zeros(label.size()[0],3,80,144).cuda()
        mask_mouth = torch.zeros(label.size()[0],3,80,144).cuda()

        #upper_lip + mouth + lower_lip
        mask_mouth_whole = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)

        for batch_index in range(0,label.size()[0]):
            mask4_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][0])-16:int(mask_list[batch_index][0])+16,int(mask_list[batch_index][1])-24:int(mask_list[batch_index][1])+24]
            mask5_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][2])-16:int(mask_list[batch_index][2])+16,int(mask_list[batch_index][3])-24:int(mask_list[batch_index][3])+24]
            mask_mouth_image[batch_index] = real_image[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]
            
            mask_mouth[batch_index] = mask_mouth_whole[batch_index,:,int(mask_list[batch_index][4])-40:int(mask_list[batch_index][4])+40,int(mask_list[batch_index][5])-72:int(mask_list[batch_index][5])+72]

        mask_mouth_image = mask_mouth * mask_mouth_image
        
        encode_label_feature = self.gen_net.forward(input_label,type="label_encoder")
        bg_feature = self.gen_net.forward(real_bg_image,type="bg_encoder")
        mask_bg = (label==0).type(torch.cuda.FloatTensor)
        mask_bg_feature = mask_bg * bg_feature
        
        loss_mask_image = 0
        loss_KL = 0

        true_samples = Variable(torch.randn(self.batch_size, self.z_dim), requires_grad=False).cuda()
        mus4, log_variances4, out4 = self.encoder_left_eye_net(mask4_image)
        variances4 = torch.exp(log_variances4 * 0.5)
        random_sample4 = Variable(torch.randn(mus4.size()).cuda(), requires_grad=True)
        correct_sample4 = random_sample4 * variances4 + mus4
        loss_KL4 = -0.5*torch.sum(-log_variances4.exp() - torch.pow(mus4,2) + log_variances4 + 1)
        reconstruce_mask4_image = self.decoder_left_eye_image_net(correct_sample4)
        loss_mask_image += self.criterionL2(reconstruce_mask4_image, mask4_image.detach()) * 10 
        mmd_loss = compute_mmd(true_samples, out4)
        loss_KL += (loss_KL4 + mmd_loss)
        decode_embed_feature4 = self.decoder_left_eye_net(correct_sample4)

        true_samples = Variable(torch.randn(self.batch_size, self.z_dim), requires_grad=False).cuda()
        mus5, log_variances5, out5 = self.encoder_right_eye_net(mask5_image)
        variances5 = torch.exp(log_variances5 * 0.5)
        random_sample5 = Variable(torch.randn(mus5.size()).cuda(), requires_grad=True)
        correct_sample5 = random_sample5 * variances5 + mus5
        loss_KL5 = -0.5*torch.sum(-log_variances5.exp() - torch.pow(mus5,2) + log_variances5 + 1)
        reconstruce_mask5_image = self.decoder_right_eye_image_net(correct_sample5)
        loss_mask_image += self.criterionL2(reconstruce_mask5_image, mask5_image.detach()) * 10 
        mmd_loss = compute_mmd(true_samples, out5)
        loss_KL += (loss_KL5 + mmd_loss)
        decode_embed_feature5 = self.decoder_right_eye_net(correct_sample5)

        true_samples = Variable(torch.randn(self.batch_size, 2048), requires_grad=False).cuda()
        mus_skin, log_variances_skin , out_skin = self.encoder_skin_net(mask_skin_image)
        variances_skin = torch.exp(log_variances_skin * 0.5)
        random_sample_skin = Variable(torch.randn(mus_skin.size()).cuda(), requires_grad=True)
        correct_sample_skin = random_sample_skin * variances_skin + mus_skin
        loss_KL_skin = -0.5*torch.sum(-log_variances_skin.exp() - torch.pow(mus_skin,2) + log_variances_skin + 1)
        reconstruce_mask_skin_image = self.decoder_skin_image_net(correct_sample_skin)
        reconstruce_mask_skin_image = mask_skin * reconstruce_mask_skin_image
        loss_mask_image += self.criterionL2(reconstruce_mask_skin_image, mask_skin_image.detach()) * 10 
        mmd_loss = compute_mmd(true_samples, out_skin)
        loss_KL += (loss_KL_skin + mmd_loss)
        decode_embed_feature_skin = self.decoder_skin_net(correct_sample_skin)        

        true_samples = Variable(torch.randn(self.batch_size, 2048), requires_grad=False).cuda()
        mus_hair, log_variances_hair, out_hair  = self.encoder_hair_net(mask_hair_image)
        variances_hair = torch.exp(log_variances_hair * 0.5)
        random_sample_hair = Variable(torch.randn(mus_hair.size()).cuda(), requires_grad=True)
        correct_sample_hair = random_sample_hair * variances_hair + mus_hair
        loss_KL_hair = -0.5*torch.sum(-log_variances_hair.exp() - torch.pow(mus_hair,2) + log_variances_hair + 1)
        reconstruce_mask_hair_image = self.decoder_hair_image_net(correct_sample_hair)
        reconstruce_mask_hair_image = mask_hair * reconstruce_mask_hair_image
        loss_mask_image += self.criterionL2(reconstruce_mask_hair_image, mask_hair_image.detach()) * 10 
        mmd_loss = compute_mmd(true_samples, out_hair)
        loss_KL += (loss_KL_hair + mmd_loss)
        decode_embed_feature_hair = self.decoder_hair_net(correct_sample_hair)

        true_samples = Variable(torch.randn(self.batch_size, 23040), requires_grad=False).cuda()
        mus_mouth, log_variances_mouth , out_mouth = self.encoder_mouth_net(mask_mouth_image)
        variances_mouth = torch.exp(log_variances_mouth * 0.5)
        random_sample_mouth = Variable(torch.randn(mus_mouth.size()).cuda(), requires_grad=True)
        correct_sample_mouth = random_sample_mouth * variances_mouth + mus_mouth
        loss_KL_mouth = -0.5*torch.sum(-log_variances_mouth.exp() - torch.pow(mus_mouth,2) + log_variances_mouth + 1)
        reconstruce_mask_mouth_image = self.decoder_mouth_image_net(correct_sample_mouth)
        reconstruce_mask_mouth_image = mask_mouth * reconstruce_mask_mouth_image
        loss_mask_image += self.criterionL2(reconstruce_mask_mouth_image, mask_mouth_image.detach()) * 10 
        mmd_loss = compute_mmd(true_samples, out_mouth)
        loss_KL += (loss_KL_mouth + mmd_loss)
        decode_embed_feature_mouth = self.decoder_mouth_net(correct_sample_mouth)
        
        loss_KL = loss_KL * 1000
        print('Losses: ', loss_KL)
        left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()

        reorder_left_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        reorder_right_eye_tensor = torch.zeros(encode_label_feature.size()).cuda()
        reorder_mouth_tensor = torch.zeros(encode_label_feature.size()).cuda()

        new_order = torch.randperm(label.size()[0])
        
        reorder_decode_embed_feature4 = decode_embed_feature4[new_order]
        reorder_decode_embed_feature5 = decode_embed_feature5[new_order]
        reorder_decode_embed_feature_mouth = decode_embed_feature_mouth[new_order]
        reorder_decode_embed_feature_skin = decode_embed_feature_skin[new_order]
        reorder_decode_embed_feature_hair = decode_embed_feature_hair[new_order]
        
        for batch_index in range(0,label.size()[0]):
            try:
                reorder_left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += reorder_decode_embed_feature4[batch_index]
                reorder_right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += reorder_decode_embed_feature5[batch_index]
                reorder_mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += reorder_decode_embed_feature_mouth[batch_index]
            except:
                print("wrong0 ! ")
                
                
        reconstruct_transfer_face = self.gen_net.forward(torch.cat((encode_label_feature,reorder_left_eye_tensor,reorder_right_eye_tensor,reorder_decode_embed_feature_skin,reorder_decode_embed_feature_hair,reorder_mouth_tensor),1),type="image_G")
        reconstruct_transfer_image = self.gen_net.forward(torch.cat((reconstruct_transfer_face,mask_bg_feature),1),type="bg_decoder")
        
        
#         parsing_label_feature = self.p_net(reconstruct_transfer_image)
#         parsing_label = softmax2label(parsing_label_feature)
        parsing_label_feature = self.p_net(reconstruct_transfer_image)
        parsing_label = softmax2label(parsing_label_feature)
        
        
        gt_label = torch.squeeze(ori_label.type(torch.cuda.LongTensor),1)
        loss_parsing = 0 #self.criterionCrossEntropy(parsing_label_feature,gt_label)*self.opt.lambda_feat
        
        
        pred_fake2_pool = self.dis_net2.forward(torch.cat((input_label, reconstruct_transfer_image.detach()), dim=1))
        loss_D2_fake = self.criterionGAN(pred_fake2_pool, False)
        # Real Detection and Loss
        # pred_real = self.discriminate(input_label, real_image)
        pred_real2 = self.dis_net2.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D2_real = self.criterionGAN(pred_real2, True)
        # GAN loss (Fake Passability Loss)        
        pred_fake2 = self.dis_net2.forward(torch.cat((input_label, reconstruct_transfer_image), dim=1))        
        loss_G2_GAN = self.criterionGAN(pred_fake2, True)
        
        
        for batch_index in range(0,label.size()[0]):
            try:
                left_eye_tensor[batch_index,:,int(mask_list[batch_index][0]/4+0.5)-4:int(mask_list[batch_index][0]/4+0.5)+4,int(mask_list[batch_index][1]/4+0.5)-6:int(mask_list[batch_index][1]/4+0.5)+6] += decode_embed_feature4[batch_index]
                right_eye_tensor[batch_index,:,int(mask_list[batch_index][2]/4+0.5)-4:int(mask_list[batch_index][2]/4+0.5)+4,int(mask_list[batch_index][3]/4+0.5)-6:int(mask_list[batch_index][3]/4+0.5)+6] += decode_embed_feature5[batch_index]
                mouth_tensor[batch_index,:,int(mask_list[batch_index][4]/4+0.5)-10:int(mask_list[batch_index][4]/4+0.5)+10,int(mask_list[batch_index][5]/4+0.5)-18:int(mask_list[batch_index][5]/4+0.5)+18] += decode_embed_feature_mouth[batch_index]
            except:
                print("wrong ! ")

        reconstruct_face = self.gen_net.forward(torch.cat((encode_label_feature,left_eye_tensor,right_eye_tensor,decode_embed_feature_skin,decode_embed_feature_hair,mouth_tensor),1),type="image_G")

        reconstruct_image = self.gen_net.forward(torch.cat((reconstruct_face,mask_bg_feature),1),type="bg_decoder")        


        # reconstruce_part image

        mask_left_eye = (label==4).type(torch.cuda.FloatTensor)
        mask_right_eye = (label==5).type(torch.cuda.FloatTensor)
        mask_mouth = ((label==7)+(label==8)+(label==9)).type(torch.cuda.FloatTensor)

        loss_L2_image = 0
        for batch_index in range(0,label.size()[0]):
            loss_L2_image += self.criterionL2( mask_left_eye*reconstruct_image, mask_left_eye*real_image) * 10 
            loss_L2_image += self.criterionL2( mask_right_eye*reconstruct_image, mask_right_eye*real_image) * 10 
            loss_L2_image += self.criterionL2( mask_skin*reconstruct_image, mask_skin*real_image) * 5 
            loss_L2_image += self.criterionL2( mask_hair*reconstruct_image, mask_hair*real_image) * 5
            loss_L2_image += self.criterionL2( mask_mouth*reconstruct_image, mask_mouth*real_image) * 10 
            loss_L2_image += self.criterionL2( reconstruct_image, real_bg_image ) * 10

        # Fake Detection and Loss
        # pred_fake_pool = self.discriminate(input_label, reconstruct_image, use_pool=True)
        pred_fake_pool = self.dis_net.forward(torch.cat((input_label, reconstruct_image.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        # Real Detection and Loss
        # pred_real = self.discriminate(input_label, real_image)
        pred_real = self.dis_net.forward(torch.cat((input_label, real_image.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        # GAN loss (Fake Passability Loss)        
        pred_fake = self.dis_net.forward(torch.cat((input_label, reconstruct_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        
        loss_G_GAN_Feat = 0
        if self.no_ganFeat_loss:
            feat_weights = 4.0 / (self.dis_n_layers + 1)
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights *                         self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.lambda_feat
        
        all_mask_tensor = torch.cat((mask_left_eye,mask_right_eye,mask_skin,mask_hair,mask_mouth),1)
        
        mask_weight_list = [10,10,5,5,10]
        # VGG feature matching loss
#         loss_G_VGG = 0
#         if self.no_vgg_loss:
#             loss_G_VGG += self.criterionVGG(reconstruct_image, real_image, all_mask_tensor, mask_weights = mask_weight_list) * self.opt.lambda_feat * 3
#             # loss_G_VGG += self.criterionVGG(reconstruct_image, real_image, mask4, weights = [1.0/4,1.0/4,1.0/4,1.0/8,1.0/8]) * self.opt.lambda_feat * 10
            
        return self.loss_filter( loss_KL,loss_mask_image,loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_L2_image, loss_parsing, loss_G2_GAN, loss_D2_real, loss_D2_fake), None if not infer else reconstruct_image, None if not infer else reconstruce_mask4_image, None if not infer else reconstruce_mask5_image, None if not infer else reconstruce_mask_skin_image, None if not infer else reconstruce_mask_hair_image, None if not infer else reconstruce_mask_mouth_image, None if not infer else reconstruct_transfer_image, None if not infer else parsing_label

