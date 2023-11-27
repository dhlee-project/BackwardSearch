import math
import time
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch

class ConditionalIRModel(nn.Module):
    def __init__(self,
                 basemodel=None,
                 basemodel_output_size=2048,
                 encoding_size=512,
                 attr_info=None,
                 args=None,
                 ):
        super().__init__()
        self.basemodel_output_size = basemodel_output_size
        self.encoding_size = encoding_size
        self.basemodel = basemodel

        self.attr_info = attr_info
        self.attr_list = self.attr_info.keys()

        if self.encoding_size != self.basemodel_output_size:
            self.encoding_embedding = True
        else:
            self.encoding_embedding = False
            print('Warning encoding_embedding is False so that eliminated decoder layer')

        # Encoder
        self.enc_linear_1 = nn.Linear(self.basemodel_output_size, self.encoding_size)
        self.enc_batchnorm_1 = nn.BatchNorm1d(self.encoding_size)
        self.enc_dropout_1 = nn.Dropout(0.1)
        self.enc_linear_2 = nn.Linear(self.encoding_size, self.encoding_size)
        # self.enc_dropout_2 = nn.Dropout(0.2)
        self.enc_batchnorm_2 = nn.BatchNorm1d(self.encoding_size)

        # Build Attr. Multi Layer Perceptron Classifier
        self.attr_clssifiers = nn.ModuleDict()
        for _attr in self.attr_list:
            _num_class = len(self.attr_info[_attr]['label2num'].keys())
            attr_clssifier = nn.Sequential(OrderedDict([
                # ('activation_1', nn.ReLU()),
                # ('batchnorm_1', nn.BatchNorm1d(self.encoding_size)),
                # ('dropout_1', nn.Dropout(0.2)),
                ('linear_1', nn.Linear(self.encoding_size, 256)),
                ('activation_2', nn.ReLU()),
                ('batchnorm_2', nn.BatchNorm1d(256)),
                ('dropout_2', nn.Dropout(0.4)),
                ('linear_2', nn.Linear(256, 128)),
                ('activation_3', nn.ReLU()),
                ('batchnorm_3', nn.BatchNorm1d(128)),
                ('dropout_3', nn.Dropout(0.4)),
                ('linear_3', nn.Linear(128, _num_class))]
            ))
            self.attr_clssifiers.update({f'{_attr}': attr_clssifier.to(args.device)})

    def forward(self, image):
        out_encoding, out_embedding = self.encoder(image)
        out_dict = {}
        out_encoding = out_encoding
        for _attr in self.attr_list:
            out_dict[_attr] = self.attr_clssifiers[_attr](out_encoding)

        out_dict['embedding'] = out_embedding
        return [out_encoding, out_dict]

    def encoder(self, image):
        embedding = self.basemodel(image)
        if self.encoding_embedding:
            x = F.relu(self.enc_linear_1(embedding))
            x = self.enc_batchnorm_1(x)
            x = self.enc_dropout_1(x)
            x = self.enc_batchnorm_2(self.enc_linear_2(x))
            encode = F.tanh(x)
            return encode, embedding
        else:
            embedding = F.tanh(self.enc_batchnorm_1(embedding))
            return embedding, embedding


class Student(nn.Module):
    def __init__(self, embedding_size, num_class):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.fc1 = nn.Linear(self.embedding_size+self.num_class, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.embedding_size)

    def forward(self, x):
        x = self.bn1(F.sigmoid(self.fc1(x)))
        x = self.bn2(F.sigmoid(self.fc2(x)))
        x = F.tanh(self.fc3(x)) # F.tanh(self.fc3(x))
        return x

class ResidualNet(nn.Module):
    def __init__(self, embedding_size, num_class):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.fc1 = nn.Linear(self.embedding_size+self.num_class, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(512, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(512, self.embedding_size)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.drop1(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.drop2(x)
        # x = self.bn3(F.relu(self.fc3(x)))
        x =  self.fc4(x)
        return x

def load_encoder(modelname='resnet50', pretrained=True, enc_update=True):
    if modelname == 'resnet50':
        model_encoder = models.resnet50(pretrained=pretrained)
        enc_output_size = model_encoder.fc.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.fc = nn.Sequential(nn.Identity())
    elif modelname == 'resnet101':
        model_encoder = models.resnet101(pretrained=pretrained)
        enc_output_size = model_encoder.fc.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.fc = nn.Sequential(nn.Identity())
    elif modelname == 'wideresnet50':
        model_encoder = models.wide_resnet50_2(weights='DEFAULT')
        enc_output_size = model_encoder.fc.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.fc = nn.Sequential(nn.Identity())
    elif modelname == 'resnext50_32x4d':
        model_encoder = models.resnext50_32x4d(pretrained=pretrained)
        enc_output_size = model_encoder.fc.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.fc = nn.Sequential(nn.Identity())
    elif modelname == 'convnext_base':
        model_encoder = models.convnext_base(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[2].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'convnext_large':
        if pretrained:
            weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
            model_encoder = models.convnext_large(weights=weights)
        else:
            model_encoder = models.convnext_large()
        enc_output_size = model_encoder.classifier[2].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'regnet_y_8gf':
        model_encoder = models.regnet_y_8gf(pretrained=pretrained)
        enc_output_size = model_encoder.fc.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.fc = nn.Sequential(nn.Identity())
    elif modelname == 'efficientnet_b3':
        model_encoder = models.efficientnet_b3(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[1].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'efficientnet_b5':
        model_encoder = models.efficientnet_b5(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[1].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'efficientnet_b0':
        model_encoder = models.efficientnet_b0(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[1].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'vgg16':
        model_encoder = models.vgg16(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[0].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Identity())
    elif modelname == 'vgg16_bn':
        model_encoder = models.vgg16_bn(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[0].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Identity())
    elif modelname == 'mobilenet_v3_large':
        model_encoder = models.mobilenet_v3_large(pretrained=pretrained)
        enc_output_size = model_encoder.classifier[0].in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'densenet121':
        pretrained = True
        model_encoder = models.densenet121(pretrained=pretrained)
        enc_output_size = model_encoder.classifier.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.classifier = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'vit_b':
        weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.DEFAULT
        model = models.vit_b_16(weights=weights)
        model_encoder = model
        enc_output_size = model_encoder.heads.head.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.heads = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'swin_b':
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        model = models.swin_b(weights=weights)
        model_encoder = model
        enc_output_size = model_encoder.head.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.head = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    elif modelname == 'regnet_y_16gf':
        weights = models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
        model = models.regnet_y_16gf(weights=weights)
        model_encoder = model
        enc_output_size = model_encoder.fc.in_features
        for param in model_encoder.parameters():
            param.requires_grad = enc_update
        model_encoder.fc = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1))
    else:
        print('Input Wrong Model Name!!, Available Model List : resnet50, ')

    return model_encoder, enc_output_size

