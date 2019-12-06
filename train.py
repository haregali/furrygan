
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import os
import time

import torch

from torch.autograd import Variable
import scipy.misc
import random

# from data.data_loader import CreateDataLoader
from furryganmodel import FurryGAN
from dataloader import DataLoader


batch_size = 1
lr = 0.0002
display_freq = 1
print_freq = 1
save_latest_freq = 1
niter_decay = 45
niter = 1
max_dataset_size = 10
niter_fix_global = 0
checkpoints_dir = '/home/prateek_jain3130/furrygan/checkpoints/'
start_epoch = 1
epoch_iter = 0
gpu_ids = [0,1]
save_epoch_freq = 10
# In[ ]:
num_epochs = 100

dataset = DataLoader(batch_size)
dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
dataset_size = len(dataset)
print(dataset_size)

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % display_freq
print_delta = total_steps % print_freq
save_delta = total_steps % save_latest_freq


# In[ ]:

    # print(torch.cuda.is_available())
print("-----LOADING MODEL-----")
furrygan = FurryGAN(checkpoints_dir, lr, niter_decay, batch_size, gpu_ids)
print("-----MODEL LOADED SUCCESSFULLY-----")

# In[ ]:


loss_mean_temp = dict()
loss_count = 0
loss_names = ['KL_embed', 'L2_mask_image', 'G_GAN','G_GAN_Feat','D_real','D_fake','L2_image','G2_GAN','D2_real','D2_fake']
for loss_name in loss_names:
    loss_mean_temp[loss_name] = 0


# In[ ]:


loss_epoch_dict = {}
error_epoch_dict = {}
    
for epoch in range(start_epoch, num_epochs):
    print("Epoch: ", epoch)
    epoch_start_time = time.time()
    # if epoch != start_epoch:
    #     epoch_iter = epoch_iter % dataset_size
    a_count = 0
    loss_iteration_dict = {}
    error_iteration_dict = {}
    for i, data in enumerate(dataset):
        prev = data
        iter_start_time = time.time()
        # total_steps += batch_size
        # epoch_iter += batch_size
        # save_fake = total_steps % display_freq == display_delta  
        losses, reconstruct, left_eye_reconstruct, right_eye_reconstruct, skin_reconstruct, hair_reconstruct, mouth_reconstruct, transfer_image, transfer_label = furrygan.forward( Variable(data['bg_image']), Variable(data['label']), data['inst'], Variable(data['image']), data['feat'], Variable(data['image_affine']), Variable(data['mask']), Variable(data['ori_label']), Variable(prev['bg_image']), Variable(prev['label']), prev['inst'], Variable(prev['image']), prev['feat'], Variable(prev['image_affine']), Variable(prev['mask']), Variable(prev['ori_label']))
        prev = data
        # losses, reconstruct = model.module.forward_vae_net(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), infer=save_fake)
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(furrygan.loss_names, losses))
#         print(loss_dict)
        a = random.random()
        loss_kl = loss_dict['KL_embed']
        #print(loss_kl)
        # loss_mask = loss_dict['L2_mask_image'] * 500
        # loss_vae_net = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)*100 + loss_dict['L2_image']*500
        
        if a_count == 1:
            a_count = 0
            a_weight = 0
        else:
            a_count = 1
            a_weight = 1
            
            
        loss_D2 = (loss_dict['D2_fake'] + loss_dict['D2_real']) * 0.5
        loss_G_together = loss_dict['G_GAN']*a_weight + loss_dict['G2_GAN'] + loss_dict['G_GAN_Feat']*a_weight + loss_dict['L2_image']*2*a_weight + loss_dict['L2_mask_image'] * 500  + loss_kl
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 * a_weight
        
        furrygan.optimizer_G_together.zero_grad()
        loss_G_together.backward()
        furrygan.optimizer_G_together.step()

        furrygan.optimizer_D.zero_grad()
        # update discriminator weights
        loss_D.backward()
        furrygan.optimizer_D.step()
    
        # update discriminator weights
        furrygan.optimizer_D2.zero_grad()
        loss_D2.backward()
        furrygan.optimizer_D2.step()
            

        # save losses to loss_mean_temp
        for loss_name in loss_names:
            loss_mean_temp[loss_name] += loss_dict[loss_name].cpu().data.numpy()
            loss_count += 1

        ############## Display results and errors ##########
        ### print out errors
#         print(loss_mean_temp)
        if i % 25 == 0:
            for loss_name in loss_names:
                loss_mean_temp[loss_name] = loss_mean_temp[loss_name].item() / loss_count

            # errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            errors = {k: v for k, v in loss_mean_temp.items()}
#             print(errors)
            t = (time.time() - iter_start_time) / batch_size
#             visualizer.print_current_errors(epoch, epoch_iter, errors, t)
#             visualizer.plot_current_errors(errors, total_steps)
            for loss_name in loss_names:
                loss_mean_temp[loss_name] = 0
            loss_count = 0
            print("Epoch: {}  Iter: {}  G_together: {}  D2: {}  D: {}".format(epoch, i*4, loss_G_together, loss_D2, loss_D))
            loss_iteration_dict[i] = {'losses':losses, 'loss_D2':loss_D2, 'loss_G_together':loss_G_together, 'loss_D':loss_D}
            error_iteration_dict[i] = errors
        # if epoch_iter >= dataset_size:
        #     break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d \t Time Taken: %d sec' %
          (epoch, time.time() - epoch_start_time))
    loss_epoch_dict[epoch] = loss_iteration_dict
    error_epoch_dict[epoch] = error_iteration_dict
    loss_iteration_dict = {}
    error_iteration_dict = {}
    ### save model for this epoch
    
    
    #TODO
    #Add save function in Furrygan
    # if epoch % 1 == 0:
    print('saving the model at the end of epoch %d' % (epoch))        
    furrygan.save('latest')
#         np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    # if epoch % save_epoch_freq == 0:
    #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
    #     furrygan.save(epoch)
#         np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')   


    ### instead of only training the local enhancer, train the entire network after certain iterations
    
    furrygan.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > niter:
        print("Updating Learning Rate")
        furrygan.update_learning_rate()
    with open('loss_dict/{}-loss_dict.pkl'.format(epoch), 'wb') as f:
        pickle.dump(loss_epoch_dict, f, pickle.HIGHEST_PROTOCOL)
    with open('error_dict/{}-error_dict.pkl'.format(epoch), 'wb') as f1:
        pickle.dump(error_epoch_dict, f1, pickle.HIGHEST_PROTOCOL)
