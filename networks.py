import numpy as np
import torch
import torch.nn as nn
from bisenet import BiSeNet
import functools


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class GeneratorNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = 64
        self.norm_type = 'batch'
        self.n_downsample = 2
        self.n_blocks_global = 9
        self.n_local_enhancers = 1
        self.n_blocks_local = 3
        self.embed_nc = 256*5
        self.padding_type='reflect'

        super(GeneratorNetwork, self).__init__()
        norm_layer = get_norm_layer(norm_type=self.norm_type)
        activation = nn.ReLU(True)
        
        downsample_model = [nn.ReflectionPad2d(3), nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0), norm_layer(self.ngf), activation]
        
        for i in range(self.n_downsample):
            mult = 2**i
            if i != self.n_downsample-1:
                downsample_model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(self.ngf * mult * 2), activation]
            else:
                downsample_model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(self.ngf * mult * 2), activation]
        self.downsample_model = nn.Sequential(*downsample_model)
        
        model=[]
        model += [nn.Conv2d(in_channels=self.ngf*(2**self.n_downsample)+self.embed_nc, out_channels=self.ngf*(2**self.n_downsample), kernel_size=1, padding=0, stride=1, bias=True)]

        mult = 2**self.n_downsample
        for i in range(self.n_blocks_global):
            model += [ResnetBlock(self.ngf * mult, padding_type=self.padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(self.n_downsample):
            mult = 2**(self.n_downsample - i)
            model += [nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0),
                       norm_layer(int(self.ngf * mult / 2)), activation]
        
        self.model = nn.Sequential(*model)

        bg_encoder = [nn.ReflectionPad2d(3), nn.Conv2d(3, self.ngf, kernel_size=7, padding=0), norm_layer(self.ngf), activation]
        self.bg_encoder = nn.Sequential(*bg_encoder)

        bg_decoder = [nn.Conv2d(in_channels=ngf*2, out_channels=self.ngf, kernel_size=1, padding=0, stride=1, bias=True)]
        bg_decoder += [nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.bg_decoder = nn.Sequential(*bg_decoder)
        
    def forward(self, input, type="label_encoder"):
        if type=="label_encoder":
            return self.downsample_model(input)
        elif type=="image_G":
            return self.model(input)
        elif type=="bg_encoder":
            return self.bg_encoder(input)
        elif type=="bg_decoder":
            # notice before bg_decoder, we should concate the feature map form G and bg_encoder
            return self.bg_decoder(input)
        else:
            print("wrong type in generator network - forward ")

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, dis_n_layers, norm_layer, use_sigoid, getIntermFeat):
        super(NLayerDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.dis_n_layers = dis_n_layers
        self.norm_layer = norm_layer
        self.use_sigoid = use_sigoid
        self.getIntermFeat = getIntermFeat
        
        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(self.input_nc, self.ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = self.ndf
        for n in range(1, self.dis_n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if self.use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if self.getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            # res = [input]
            # for n in range(self.n_layers+2):
            #     model = getattr(self, 'model'+str(n))
            #     res.append(model(res[-1]))
            # return res[1:]
            res = [input]
            for n in range(self.dis_n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            print("debug in networks line 721 ----")
            print(len(res[-2:]))
            return res[-2:]
        else:
            return self.model(input)
        
        
class DiscriminatorNetwork(nn.Module):
    def __init__(self, input_nc, dis_n_layers, numD, use_sigmoid):
        super(DiscriminatorNetwork, self).__init__()
        self.input_nc = input_nc
        self.dis_n_layers = dis_n_layers
        self.num_D = num_D
        self.use_sigmoid = use_sigmoid
        self.ndf = 64
        self.norm_type = 'batch'
        self.getIntermFeat = True
        
        norm_layer = get_norm_layer(norm_type=self.norm_type)
 
        for i in range(self.num_D):
            netD = NLayerDiscriminator(self.input_nc, self.ndf, self.dis_n_layers, norm_layer, self.use_sigmoid, self.getIntermFeat)
            if self.getIntermFeat:                                
                for j in range(self.dis_n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[-2:]
        else:
            return [model(input)]
        
    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.dis_n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            # print("i is ")
            # print(i)
            # print("input_downsampled size is ")
            # print(input_downsampled.size())
            
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        
        global printlayer_index
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1,output_padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            # printlayer = [PrintLayer(name = str(printlayer_index))]
            # printlayer_index += 1
            # model = printlayer + down + [submodule] + up
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = printlayer + down + up
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias,output_padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # printlayer = [PrintLayer(str(printlayer_index))]
            # printlayer_index += 1
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
                # model = printlayer + down + [submodule] + printlayer + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule]  + up
                # model = printlayer + down + [submodule] + printlayer + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        model_output = self.model(x)
        wb,hb = model_output.size()[3],model_output.size()[2]
        wa,ha = x.size()[3],x.size()[2]
        l = int((wb-wa)/2)
        t = int((hb-ha)/2)
        model_output = model_output[:,:,t:t+ha,l:l+wa]
        if self.outermost:
            return model_output
        else:
            return torch.cat([x, model_output], 1)           #if not the outermost block, we concate x and self.model(x) during forward to implement unet
    
class UnetGenerator(nn.Module):
    def __init__(self, segment_classes, input_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        output_nc = segment_classes
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        #maybe do some check here with softmax
        self.model = unet_block

    def forward(self, input):
        softmax = torch.nn.Softmax(dim = 1)
        return softmax(self.model(input))    

# class PNetwork(nn.Module):
#     def __init__(self, label_nc, output_nc):
#         self.label_nc = label_nc
#         self.output_nc = output_nc
#         self.ngf = 64
#         self.norm_type = 'batch'
#         self.use_dropout = True
#         norm_layer = get_norm_layer(norm_type=norm)
        
#         netP = UnetGenerator(self.label_nc, self.input_nc, 6, self.ngf, norm_layer=norm_layer, use_dropout=self.use_dropout)



class PNetwork(nn.Module):
    def __init__(self, model_path="../faceparsing/res/cp/face_parsing.pth"):
        self.net = BiSeNet(n_classes=n_classes)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def mask_model(self, image_tensor):
        with torch.no_grad():
            img = torch.unsqueeze(image_tensor, 0)
            out = self.net(image_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_parsing_anno = np.where(vis_parsing_anno == 2, 100, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 3, 2, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 100, 3, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 4, 500, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 5, 4, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 500, 5, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 6, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 7, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 8, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 9, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 10, 6, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 11, 8, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 12, 7, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 13, 9, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 14, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 15, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 16, 0, vis_parsing_anno)
            vis_parsing_anno = np.where(vis_parsing_anno == 17, 10, vis_parsing_anno)
            
            for i in range(18, 50):
                vis_parsing_anno = np.where(vis_parsing_anno == i, 0, vis_parsing_anno)

            return torch.from_numpy(vis_parsing_anno).cuda()
       

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
        
        
# class MMDEncoderBlock(nn.Module):
#     def __init__(self):
#         super(MMDEncoderBlock, self).__init()
#         self.conv1 = nn.Conv2d(1,64,4,2, padding=1)
#         self.relu1 = nn.LeakyReLU()
#         self.conv2 = nn.Conv2d(64,128,4,2, padding=1)
#         self.relu2 = nn.LeakyReLU()
#         self.flat = Flatten()
#         self.linear1 = nn.Linear(6272, 1024)
#         self.relu3 = nn.LeakyReLU()
#         self.linear2 = nn.Linear(1024, 512)
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.flat(x)
#         x = self.linear1(x)
#         x = self.relu3(x)
#         x = self.linear2(x)
#         return x
        
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=7, padding=3, stride=4):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, padding=padding, stride=stride)
#         self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        self.bn = nn.BatchNorm2d(num_features=channel_out)
#         self.relu = nn.ReLU(True)
        self.relu = nn.LeakyReLU(True)

    def forward(self, ten, out=False,t = False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = self.relu(ten)
            return ten
        

class  EncoderGenerator_mask_skin(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_mask_skin, self).__init__()
        layers_list = []
        # 3*256*256
        
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=3, padding=1, stride=1))  # 64*256*256
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=3, padding=1, stride=1))  # 128*256*256
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=3, padding=1, stride=2))  # 256*128*128
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*64*64
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*32*32
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*16*16
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*8*8
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*4*4
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*2*2
        # final shape Bx128*4*4
        self.conv = nn.Sequential(*layers_list)

        # self.c_mu = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        # self.c_var = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512*2*2, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512*2*2, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(True),
                                nn.Linear(in_features=1024, out_features=512))

    def forward(self, ten):
        ten_org = self.conv(ten)
        ten = ten.view(ten_org.size()[0],-1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu,logvar, ten_org

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_skin, self).__call__(*args, **kwargs)
    
class  EncoderGenerator_mask_mouth(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_mask_mouth, self).__init__()
        layers_list = []
        
        # 3*80*144
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=3, padding=1, stride=1))  # 64*80*144
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=3, padding=1, stride=1))  # 128*80*144
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=3, padding=1, stride=2))  # 256*40*72
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*20*36
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*10*18
        layers_list.append(EncoderBlock(channel_in=512, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*5*9
        
        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512*5*9, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512*5*9, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        # self.c_mu = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        # self.c_var = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)


    def forward(self, ten):
        ten_org = self.conv(ten)
        ten = ten.view(ten_org.size()[0],-1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu,logvar, ten_org

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_mouth, self).__call__(*args, **kwargs)

class  EncoderGenerator_mask_eye(nn.Module):
    """docstring for  EncoderGenerator"""
    def __init__(self, norm_layer):
        super( EncoderGenerator_mask_eye, self).__init__()
        layers_list = []
        
        # 3*32*48
        layers_list.append(EncoderBlock(channel_in=3, channel_out=64, kernel_size=3, padding=1, stride=1))  # 64*32*48
        layers_list.append(EncoderBlock(channel_in=64, channel_out=128, kernel_size=3, padding=1, stride=2))  # 128*16*24
        layers_list.append(EncoderBlock(channel_in=128, channel_out=256, kernel_size=3, padding=1, stride=2))  # 256*8*12
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*4*6
        layers_list.append(EncoderBlock(channel_in=256, channel_out=512, kernel_size=3, padding=1, stride=2))  # 512*2*3
        
        # final shape Bx256*7*6
        self.conv = nn.Sequential(*layers_list)
        self.fc_mu = nn.Sequential(nn.Linear(in_features=512*2*3, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        self.fc_var = nn.Sequential(nn.Linear(in_features=512*2*3, out_features=1024),
                                # nn.BatchNorm1d(num_features=1024,momentum=0.9),
                                nn.LeakyReLU(True),
                                nn.Linear(in_features=1024, out_features=512))
        # self.c_mu = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)
        # self.c_var = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)


    def forward(self, ten):
        ten_org = self.conv(ten)
        ten = ten.view(ten_org.size()[0],-1)
        mu = self.fc_mu(ten)
        logvar = self.fc_var(ten)
        return mu,logvar, ten_org

    def __call__(self, *args, **kwargs):
        return super(EncoderGenerator_mask_eye, self).__call__(*args, **kwargs)
    
        
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=4, padding=1, stride=2, output_padding=0, norelu=False):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        # self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding)
        # self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        # self.bn = nn.InstanceNorm2d(channel_out, momentum=0.9,track_running_stats=True)
        layers_list = []
        layers_list.append(nn.ConvTranspose2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=output_padding))
        layers_list.append(nn.BatchNorm2d(channel_out, momentum=0.9))
        if norelu == False:
            layers_list.append(nn.LeakyReLU(True))
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.conv(ten)
        return ten

class DecoderGenerator_mask_skin_image(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_skin_image, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*2))
        # input is 512*2*2
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*4
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*8*8
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*16*16
        layers_list.append(DecoderBlock(channel_in=512, channel_out=512, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*32*32
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0))  #128*64*64
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0))  #64*128*128
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0))  #64*256*256
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,3,kernel_size=5,padding=0))
        layers_list.append(nn.Tanh())
        
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator_mask_skin, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 2)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 256
        assert ten.size()[3] == 256
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_skin_image, self).__call__(*args, **kwargs)
    
class DecoderGenerator_mask_mouth_image(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_mouth_image, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512*4*4),
        #                         nn.BatchNorm1d(num_features=512*4*4, momentum=0.9),
        #                         nn.ReLU(True))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*5*9))
        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #10*18
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0)) #20*36
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #40*72
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #80*144
        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #96*160
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,3,kernel_size=5,padding=0))
        layers_list.append(nn.Tanh())

        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*12*14

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 5, 9)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 80
        assert ten.size()[3] == 144
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_mouth_image, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_eye_image(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_eye_image, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512*4*4),
        #                         nn.BatchNorm1d(num_features=512*4*4, momentum=0.9),
        #                         nn.ReLU(True))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*3, bias=False))
        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2, output_padding=0)) #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=128, kernel_size=4, padding=1, stride=2, output_padding=0)) #128*8
        layers_list.append(DecoderBlock(channel_in=128, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #64*16
        layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #64*32
        # layers_list.append(DecoderBlock(channel_in=64, channel_out=64, kernel_size=4, padding=1, stride=2, output_padding=0)) #64*64
        layers_list.append(nn.ReflectionPad2d(2))
        layers_list.append(nn.Conv2d(64,3,kernel_size=5,padding=0))
        layers_list.append(nn.Tanh())

        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*12*14

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 3)
        ten = self.conv(ten)
        assert ten.size()[1] == 3
        assert ten.size()[2] == 32
        assert ten.size()[3] == 48
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_eye_image, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_mouth(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_mouth, self).__init__()
        

        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*5*9))
        layers_list = []

        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2)) #10*18
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #20*36
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #40*72

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # ten = self.fc(ten)
        # ten = ten.view(ten.size()[0],512, 4, 4)
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 5, 9)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 20
        assert ten.size()[3] == 36
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_mouth, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_eye(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_eye, self).__init__()
        # start from B*1024
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512*4*4),
        #                         nn.BatchNorm1d(num_features=512*4*4, momentum=0.9),
        #                         nn.ReLU(True))
        # self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=256*6*7, bias=False))
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*3, bias=False))
        layers_list = []
        # layers_list.append(nn.BatchNorm2d(256, momentum=0.9))
        # layers_list.append(nn.ReLU(True))
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2)) #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #256*8
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2)) #256*16
        # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1)) #256*16
        # # layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=3, padding=1, stride=1, output_padding=0)) #256*12*14

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator, print some shape ")
        # ten = self.fc(ten)
        # ten = ten.view(ten.size()[0],512, 4, 4)
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 3)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 8
        assert ten.size()[3] == 12
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_eye, self).__call__(*args, **kwargs)


class DecoderGenerator_mask_skin(nn.Module):
    def __init__(self, norm_layer):  
        super(DecoderGenerator_mask_skin, self).__init__()
        # input is 128*4*4
        self.fc = nn.Sequential(nn.Linear(in_features=512, out_features=512*2*2))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=512, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*4
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*8
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*16
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*32
        layers_list.append(DecoderBlock(channel_in=256, channel_out=256, kernel_size=4, padding=1, stride=2))  #256*64
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        # print("in DecoderGenerator_mask_skin, print some shape ")
        ten = self.fc(ten)
        ten = ten.view(ten.size()[0],512, 2, 2)
        ten = self.conv(ten)
        assert ten.size()[1] == 256
        assert ten.size()[2] == 64
        return ten

    def __call__(self, *args, **kwargs):
        return super(DecoderGenerator_mask_skin, self).__call__(*args, **kwargs)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer