import torch
import torch.nn as nn
import torchvision
from .ops import ConsensusModule, Identity
import numpy as np 
import os

from dataset.transforms import GroupMultiScaleCrop,GroupRandomHorizontalFlip
class TSN(nn.Module):
    def __init__(self, num_class,num_segments,modality,
                base_model='resnet101',new_length=None,
                consensus_type='avg',before_softmax=True,
                dropout=0.8,crop_num=1,partial_bn=True
                ):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.partial_bn=partial_bn
        self.consensus_type = consensus_type
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(
        """
        Initializing TSN with base model: {}.
        TSN Configurations:
            input_modality:     {}
            num_segments:       {}
            new_length:         {}
            consensus_module:   {}
            dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout))

        self._prepare_base_model(base_model)
        feature_dim=self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self._construct_model(2)
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self._construct_model(3)

        self.consensus=ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax=nn.Softmax()


    def forward(self,input):
        #(batch_size,segment*len*3,h,w)

        sample_len=(3 if self.modality=='RGB' else 2)*self.new_length

        if self.modality=='RGBDiff':
            sample_len=3*self.new_length
            input=self._get_diff(input)

        base_out=self.base_model(input.view((-1,sample_len)+input.size()[-2:]))
        #input shape:(batch*seg,len*3,h,w) output shape:(batch*seg,num_classes)
        
        if self.dropout>0:
            base_out=self.new_fc(base_out)
        if not self.before_softmax:
            base_out=self.softmax(base_out)

        base_out=base_out.view((-1,self.num_segments)+base_out.size()[1:])
        output=self.consensus(base_out)

        return output.squeeze()#(batch*num_classes)

    def _prepare_base_model(self,base_model):
        #返回基础网络结构
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model=getattr(torchvision.models,base_model)(pretrained=True)
            self.base_model.last_layer_name='fc'
            self.input_size=224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality=='Flow':
                self.input_mean=[0.5]
                self.input_std=[np.mean(self.input_std)]
            elif self.modality=='RGBDiff':
                self.input_mean = self.input_mean+ [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self,num_class):
        #返回输入特征的维度，也就是原网络最后一层的输入
        #重构最后一层，并且重新初始化最后一层的参数
        #并根据是否设置dropout来判断是否改变原网络结构

        feature_dim=getattr(self.base_model,self.base_model.last_layer_name).in_features
        if self.dropout!=0:
            setattr(self.base_model,self.base_model.last_layer_name,nn.Dropout(p=self.dropout))
            self.new_fc=nn.Linear(feature_dim,num_class)
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None

        std=0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            nn.init.normal_(self.new_fc.weight, 0, std)
            nn.init.constant_(self.new_fc.bias, 0)
        return feature_dim

    def _construct_model(self,channels):
        #把第一层卷积原本是处理图像输入3通道的改成能处理2*length通道
        modules=list(self.base_model.modules())
        first_conv_idx=list(filter(lambda x:isinstance(modules[x],nn.Conv2d),list(range(len(modules)))))[0]
        conv_layer=modules[first_conv_idx]#第一个卷积层
        container=modules[first_conv_idx-1]#包含第一个卷积层的整个子网络
        #print(container)

        params=[x.clone() for x in conv_layer.parameters()]
        kernel_size=params[0].size()  #卷积核的维度c2*c1*w*h
        new_kernel_size=kernel_size[:1]+(channels*self.new_length,)+kernel_size[2:]#optiecal flow个数*2(x,y两个方向) #RGBDiff 3个通道
        new_kernel=params[0].data.mean(dim=1,keepdim=True).expand(new_kernel_size).contiguous()#取平均之后重复

        new_conv=nn.Conv2d(channels*self.new_length,conv_layer.out_channels,
                            conv_layer.kernel_size,conv_layer.stride,conv_layer.padding,
                            bias=True if len(params)==2 else False)
        new_conv.weight.data=new_kernel
        if len(params)==2:
            new_conv.bias.data=params[1].data
        layer_name=list(container.state_dict().keys())[0][:-7]
        setattr(container,layer_name,new_conv)

    def train(self,mode=True):
        super(TSN,self).train(mode)
        count=0
        if self.partial_bn:
            print('Freezing BatchNorm2D except the first one.')
            for m in self.base_model.modules():
                if isinstance(m,nn.BatchNorm2d):
                    count+=1
                    if count>=2:
                        m.eval()
                        m.weight.requires_grad=False
                        m.bias.requires_grad=False

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size*256//224


    def _get_diff(self,input):
        #get n RGBdiff from  n+1 RGB
        #(batch,seg*len*3,h,w)
        input_c=3
        input_view=input.view((-1,self.num_segments,self.new_length+1,input_c,)+input.size()[2:])#(batch,seg,len,3,h,w)
        new_data=input_view[:,:,1:,:,:,:].clone()

        for x in reversed(list(range(1,self.new_length+1))):
            new_data[:,:,x-1,:,:,:]=input_view[:,:,x:,:,:]-input_view[:,:,x-1,:,:,:]

        return new_data   

    def get_augmentation(self):
        if self.modality=='RGB':
            return torchvision.transforms.Compose([
                    GroupMultiScaleCrop(self.input_size,[1,.875,.75,.66]),
                    GroupRandomHorizontalFlip(is_flow=False)
                    ])         
        elif self.modality=='Flow':
            return torchvision.transforms.Compose([
                    GroupMultiScaleCrop(self.input_size,[1,.875,.75]),
                    GroupRandomHorizontalFlip(is_flow=True)
                    ])  
        elif self.modality=='RGBDiff':
            return torchvision.transforms.Compose([
                    GroupMultiScaleCrop(self.input_size,[1,.875,.75]),
                    GroupRandomHorizontalFlip(is_flow=False)
                    ])  

    def get_optim_policies(self):
        #对不同的参数设置不同的学习率
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self.partial_bn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 
             'name': "BN scale/shift"},
        ]

if __name__ == '__main__':
    pass
    
    #tsn=TSN(num_class=101,num_segments=3,modality='Flow')
    #print(tsn.get_optim_policies())
    #print(tsn.get_augmentation())