#!/usr/bin/env python
# coding: utf-8

# # First glance and etc

# In[1]:


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from IPython.display import HTML
import math
from PIL import Image
import dlib
import matplotlib.pyplot as plt


# In[2]:


device = torch.device('cuda:0')


# ## Data

# #### Constants

# In[3]:


IMG_WHT = 160
IMG_HHT = 160
ETA = 1e-2


# # Layers
# 

# In[4]:


def weigth_initialization(conv_weigths, init):
    if (init == 'xavier_normal'):
        nn.init.xavier_normal_(conv_weigths)
    if (init == 'xavier_uniform'):
        nn.init.xavier_uniform_(conv_weigths)
    return
        

def conv(input_channels, 
         output_channels,
         kernel_size, stride, 
         padding, weigth_init = 'xavier_normal', 
         batch_norm = False,
         activation=nn.ReLU()):
    a = []
    a.append(nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    weigth_initialization(a[-1].weight, weigth_init)
    if activation is not None:
        a.append(activation)
    if batch_norm:
        a.append(nn.BatchNorm2d(output_channels))
    return nn.Sequential(*a)

def deconv(input_channels, 
         output_channels,
         kernel_size, stride, 
         padding, weigth_init = 'xavier_normal', 
         batch_norm = False,
         activation=nn.ReLU()):
    a = []
    a.append(nn.ConvTranspose2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    weigth_initialization(a[-1].weight, weigth_init)
    if activation is not None:
        a.append(activation)
    if batch_norm:
        a.append(nn.BatchNorm2d(output_channels))
    return nn.Sequential(*a)


# In[5]:


class ResidualBlock(nn.Module):
    def __init__(self, input_channels,
                output_channels = None, 
                kernel_size = 3, stride = 1, 
                padding = None, weight_init = 'xavier_normal', 
                batch_norm = False,
                activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        if output_channels is None:
            output_channels = input_channels // stride
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.conv(input_channels, output_channels, 1, stride, 0, None, False, None)
            
        a = []
        a.append( conv( input_channels , input_channels  , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , weight_init ,  False, activation))
        a.append( conv( input_channels , output_channels , kernel_size , 1 , padding if padding is not None else (kernel_size - 1)//2 , None , False, None))
        self.model = nn.Sequential(*a)
        
    def forward(self, x):
        return self.activation(self.model(x) + self.shortcut(x))
    


# # Formula to calculate padding (for me)

# output_size: $$O = \Big[\frac{W−F+2P}{S}\Big]+1$$ where W - input_size, F - filter size, P - padding, S - stride $$$$
# Therefore, $$P = \frac{S \cdot (O - 1) + F - W}{2} $$
# 

# In[6]:



def calc_conv_outsize(input_size, filter_size, stride, pad):
    return math.floor((input_size - filter_size + 2 * pad) / stride) + 1


# In[7]:


def round_half_up(n):
    return math.floor(n + 0.5)


# In[8]:


def calc_conv_pad(input_size, output_size, filter_size, stride):
    return round_half_up((stride * (output_size - 1) + filter_size - input_size) / 2)

def calc_deconv_pad(input_size, output_size, filter_size, stride):
    return round_half_up((stride * (input_size - 1) + filter_size - output_size) / 2)


# ## Generator

# In[9]:


def same_padding(size, kernel_size, stride, dilation):
    return ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2


# In[104]:


def show_feature_map(c):
    s = int(c.size()[1] / 4)
    fig, ax = plt.subplots(s, 4, figsize=(15, 10))
    for i in range(s):
        for j in range(4):
            ax[i, j].imshow(c[0][j*s + i].cpu().detach())
    
def show(tensor):
    img = transforms.ToPILImage()(tensor)
    plt.imshow(img, cmap='gray')
    plt.show()


# In[150]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        ##ENCODER##

        
        f = np.array([64, 64, 128, 256, 512]) / 2
        
        f = f.astype(int)

        
        batch_norm = False
        self.conv0 = nn.Sequential(conv(5, f[0], 7, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 7, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 128 x 128
                                   ResidualBlock(f[0], activation = nn.ReLU(ETA))) 
        self.conv1 = nn.Sequential(conv(f[0], f[1], 5, 2, calc_conv_pad(IMG_WHT, IMG_WHT / 2, 5, 2), "xavier_normal",  batch_norm, nn.ReLU(ETA)), # 64 x 64
                                   ResidualBlock(f[1], activation = nn.ReLU(ETA)))
        self.conv2 = nn.Sequential(conv(f[1], f[2], 3, 2, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 4, 3, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)),  # 32 x 32
                                   ResidualBlock(f[2], activation = nn.ReLU(ETA))) 
        self.conv3 = nn.Sequential(conv(f[2], f[3], 3, 2, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 8, 3, 2), "xavier_normal",  batch_norm, nn.ReLU(ETA)), # 16 x 16
                                   ResidualBlock(f[3], activation = nn.ReLU(ETA))) 
        self.conv4 = nn.Sequential(conv(f[3], f[4], 3, 2, calc_conv_pad(IMG_WHT / 8, IMG_WHT / 16, 3, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 8 x 8
                                   ResidualBlock(f[4], activation = nn.ReLU(ETA))) 
      
        self.fc1 = nn.Linear(f[4] * 10 * 10, f[4])
        self.relu = nn.ReLU(inplace=True)
        self.maxout = nn.MaxPool1d(2 )
        ##DECODER##
        
        self.fc2 = nn.Linear(f[3], f[1] * 10 * 10) 
        
        #first path - 3 deconvs
        
        
        f = np.array([64, 32, 16, 8]) /2
        f = f.astype(int)

        
        self.dc0_1 = deconv(f[0], f[1], 4, 4, calc_deconv_pad(IMG_WHT / 16, IMG_WHT / 4, 4, 4), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)) # 32 x 32
        self.dc0_2 = deconv(f[1], f[2], 2, 2, calc_deconv_pad(IMG_WHT / 4, IMG_WHT / 2, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)) # 64 x 64
        self.dc0_3 = deconv(f[2], f[3], 2, 2, calc_deconv_pad(IMG_WHT / 2, IMG_WHT, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)) # 128 x 128
        
        #u-net path - 4 deconvs

        
        f = np.array([512, 256, 128, 64, 32, 16, 8]) / 2
        f = f.astype(int)

        
        self.dc1 = nn.Sequential(deconv(f[0] + f[3], f[0], 2, 2, calc_deconv_pad(IMG_WHT / 16, IMG_WHT / 8, 2, 2),"xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 16 x 16
                                 ResidualBlock(f[0], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[0], activation = nn.ReLU(ETA)))
        self.dc2 = nn.Sequential(deconv(f[0] + f[1], f[1], 2, 2, calc_deconv_pad(IMG_WHT / 8, IMG_WHT / 4, 2, 2),"xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 32 x 32
                                 ResidualBlock(f[1], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[1], activation = nn.ReLU(ETA)))
        self.dc3 = nn.Sequential(deconv(f[2] + f[1] + 3 + f[4], f[2], 2, 2, calc_deconv_pad(IMG_WHT / 4, IMG_WHT / 2, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 64 x 64
                                 ResidualBlock(f[2], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[2], activation = nn.ReLU(ETA)))
        self.dc4 = nn.Sequential(deconv(f[2] + f[3] + 3 + f[5], f[3], 2, 2, calc_deconv_pad(IMG_WHT / 2, IMG_WHT, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 128 x 128
                                 ResidualBlock(f[3], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[3], activation = nn.ReLU(ETA)))

        #final convs
        
        self.conv5 = conv(f[1], 3, 3, 1, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 4, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 32 x 32
        self.conv6 = conv(f[2], 3, 3, 1, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 2, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 64 x 64
        self.conv7 = conv(f[3] + f[3] + 3 + f[6], f[3], 5, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 5, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 128 x 128
        self.conv8 = conv(f[3], f[4], 3, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 128 x 128
        self.conv9 = conv(f[4], 3, 3, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 128 x 128
        
    def forward(self, picture, landmarks_real, landmarks_wanted): #img = x
        #Ecoder
        x = torch.cat([picture, landmarks_real, landmarks_wanted], dim = 1)
        c0 = self.conv0(x)
        c1 = self.conv1(c0)
        #show_feature_map(c0)
        c2 = self.conv2(c1)

        c3 = self.conv3(c2)

        c4 = self.conv4(c3)
    
        tmp = self.num_flat_features(c4)
        f1 = c4.view(x.size()[0], tmp)
        f1 = self.fc1(f1)
        f1 = self.relu(f1)
        f1 = f1.unsqueeze(0)
        maxout = self.maxout(f1)[0]
        
        #Decoder
        #1
        
        f2 = self.fc2(maxout)
        rsh = f2.reshape((x.size()[0], int(64 / 2), 10, 10))
        
        dc01 = self.dc0_1(rsh)
        
        dc02 = self.dc0_2(dc01)
        dc03 = self.dc0_3(dc02)
        
        #2
        dc1r = self.dc1(torch.cat((rsh, c4), dim=1))
        dc2r = self.dc2(torch.cat((dc1r, c3), dim=1))
        pic_div_2 = nn.MaxPool2d(2)(picture)
        pic_div_4 = nn.MaxPool2d(2)(pic_div_2)
        dc3r = self.dc3(torch.cat((dc2r, c2, pic_div_4, dc01), dim=1))
        dc4r = self.dc4(torch.cat((dc3r, c1, pic_div_2, dc02), dim=1))
        #3
        
        c5 = self.conv5(dc2r)
        c6 = self.conv6(dc3r)
    
        c7 = self.conv7(torch.cat((dc4r, c0, picture, dc03), dim=1))
        #show_feature_map(c7)
        c8 = self.conv8(c7)
        #show_feature_map(c8)
        c9 = self.conv9(c8)
   
        
        return c5, c6, c9  #img_32, img_64, img_128
        #return picture, nn.MaxPool2d(2)(picture), nn.MaxPool2d(2)(nn.MaxPool2d(2)(picture))
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[151]:


from modelsummary import summary


# In[152]:



# ## Disciminator

# In[156]:

class Discriminator_faces(nn.Module):
    def __init__(self):
        super(Discriminator_faces, self).__init__()
        batch_norm = False
        self.model = nn.Sequential(conv(6, 64, 4, 2, calc_conv_pad(IMG_WHT, IMG_WHT / 2, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 64 x 64
                                   conv(64, 128, 4, 2, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 4, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 32 x 32
                                   conv(128, 256, 4, 2, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 8, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 16 x 16
                                   conv(256, 512, 4, 2, calc_conv_pad(IMG_WHT / 8, IMG_WHT / 16, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 8 x 8
                                   conv(512, 512, 4, 1, calc_conv_pad(IMG_WHT / 16, 7, 4, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 7 x 7
                                   conv(512, 1, 4, 1, calc_conv_pad(7, 6, 4, 1), "xavier_normal", False, None)) # 6 x 6
    def forward(self, x):
        return self.model(x)
    
class Discriminator_marks(nn.Module):
    def __init__(self):
        super(Discriminator_marks, self).__init__()
        batch_norm = False
        self.model = nn.Sequential(conv(4, 64, 4, 2, calc_conv_pad(IMG_WHT, IMG_WHT / 2, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 64 x 64
                                   conv(64, 128, 4, 2, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 4, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 32 x 32
                                   conv(128, 256, 4, 2, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 8, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 16 x 16
                                   conv(256, 512, 4, 2, calc_conv_pad(IMG_WHT / 8, IMG_WHT / 16, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 8 x 8
                                   conv(512, 512, 4, 1, calc_conv_pad(IMG_WHT / 16, 7, 4, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 7 x 7
                                   conv(512, 1, 4, 1, calc_conv_pad(7, 6, 4, 1), "xavier_normal", False, None)) # 6 x 6
    def forward(self, x):
        return self.model(x)
    


# # Data

# In[17]:


multipie_path = r'multi_PIE_crop_128/001/001_01_01_051_00_crop_128.png'


# In[18]:


import shutil
import os


# ### НЕ ЗАПУСКАТЬ, ЕСЛИ  папки img_list и frontal_faces УЖЕ ЕСТЬ

# In[19]:


# ### Train data generator

# In[128]:


import cv2
from imutils import face_utils
from torch.utils.data import Dataset


# In[129]:


# ПОДГОТОВИЛИ DLIB
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LAND_MARKS_INDEXES = [37, 40, 43, 46, 32, 36, 49, 52, 55, 58]


# In[130]:


IMG_LIST_PATH = 'img_list_full'
FRONTAL_FACES_PATH = 'frontal_faces_full'


# In[131]:


def get_landmarks(link):
        img = cv2.imread(link)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # The square where we place landmarks
        black = cv2.imread("black_square160.png")
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (i, (x, y)) in enumerate(shape):
                if ((i + 1) in LAND_MARKS_INDEXES):
                    cv2.circle(black, (x, y), 2, (0, 255, 0), -1)
        return transforms.ToTensor()(Image.fromarray(black))[1].unsqueeze(0)


class TrainDataset( Dataset):
    def __init__( self , img_list ):
        super(type(self),self).__init__()
        self.img_list = img_list
    def __len__( self ):
        return len(self.img_list)
    def __getitem__( self , idx ):
        #example - 001_01_01_010_05_crop_128
        batch = {}
        img_name = self.img_list[idx]
        sp = img_name.split('_')
        img_frontal_name = '_'.join(sp[:3]) + '_051_' + sp[4] + '_crop_'
        with Image.open('/'.join([IMG_LIST_PATH, img_name])) as img:
            batch['img'] = transforms.ToTensor()(img)
        with Image.open( '/'.join([FRONTAL_FACES_PATH, img_frontal_name + '160.png'])) as img:
            batch['img160_wanted'] = transforms.ToTensor()(img)
        with Image.open( '/'.join([FRONTAL_FACES_PATH, img_frontal_name + '40.png'])) as img:
            batch['img40_wanted'] = transforms.ToTensor()(img)
        with Image.open( '/'.join([FRONTAL_FACES_PATH, img_frontal_name + '80.png'])) as img:
            batch['img80_wanted'] = transforms.ToTensor()(img)
        # GET LANDMARKS

        batch['landmarks_real'] = get_landmarks('/'.join([IMG_LIST_PATH, img_name]))
        batch['landmarks_wanted'] = get_landmarks('/'.join([FRONTAL_FACES_PATH, img_frontal_name + '160.png']))
        
        return batch


# ### Составим img_list из фоток с поворотом не больше чем на +- 60 градусов

# In[132]:


img_list = []
for f in os.listdir(IMG_LIST_PATH):
    img_list.append(f)
        


# ##### Small dataset for checking 

# In[133]:


# img_list = img_list[0:200]


# ##### Shuffle and divide into train and test

# In[134]:


random_seed = 42
val_split = 0.3


# In[135]:


indices = list(range(0, len(img_list)))
split = int(np.floor(val_split * len(img_list)))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# In[136]:


train_img_list = list(np.array(img_list)[train_indices])
test_img_list = list(np.array(img_list)[val_indices[:400]])


# In[137]:


train = TrainDataset(train_img_list)
test = TrainDataset(test_img_list)




# In[140]:


train_dataloader = torch.utils.data.DataLoader( train , batch_size = 4 , shuffle = True) 
test_dataloader = torch.utils.data.DataLoader( test , batch_size = 4 , shuffle = True) 


# # Feature extractor 
# 

# In[141]:


from keras.models import load_model
from PIL import Image
import torchvision.transforms.functional as ttf


# In[142]:


from facenet_pytorch import InceptionResnetV1


# # Loss

# In[143]:


def set_requires_grad(module , b ):
    for parm in module.parameters():
        parm.requires_grad = b


# In[144]:





# ## Tensorboard

# In[145]:


from torch.utils.tensorboard import SummaryWriter


# # LEARNING

# In[146]:


def save_model(model,model_name, dirname,epoch):
    if type(model).__name__ == torch.nn.DataParallel.__name__:
        model = model.module
    torch.save( model.state_dict() , '{}/{}_epoch{}.pth'.format(dirname,type(model).__name__ + '_' + model_name,epoch ) )
    
def save_optimizer(optimizer,model, model_name, dirname,epoch):
    if type(model).__name__ ==  torch.nn.DataParallel.__name__:
        model = model.module
    torch.save( optimizer.state_dict() , '{}/{}_epoch_{}.pth'.format(dirname,type(optimizer).__name__ +'_' +type(model).__name__ + '_' + model_name,epoch ) )

def resume_model(model,model_name, path ,i, strict= True):
    if type(model).__name__ == torch.nn.DataParallel.__name__:
        model = model.module
    p = "{}/{}_epoch{}.pth".format( path,type(model).__name__+ '_' + model_name,i )
    print(p)
    if os.path.exists( p ):
        model.load_state_dict(  torch.load( p ) , strict = strict)
        return i
    else:
        raise NameError("No model saved file with this epoch")
def resume_model_old(model, path ,i, strict= True):
    if type(model).__name__ == torch.nn.DataParallel.__name__:
        model = model.module
    p = "{}/{}_epoch{}.pth".format( path,type(model).__name__,i )
    print(p)
    if os.path.exists( p ):
        model.load_state_dict(  torch.load( p ) , strict = strict)
        return i
    else:
        raise NameError("No model saved file with this epoch")


# In[147]:


LAST_EPOCH = 0

RESUME_MODEL = False

LEARNING_RATE = 0.0002



# In[158]:


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) # предобученная нейросеть
for param in resnet.parameters():
        param.requires_grad = False
        
G = nn.DataParallel(Generator()).to(device)
D1 = nn.DataParallel(Discriminator_faces()).to(device) # Этот дискриминатор отличает изображение (img) и то, как генератор его повернул 
D2 = nn.DataParallel(Discriminator_marks()).to(device) # Этот дискриминатор отличает повернутое G изображение и landmarks_wanted 

if RESUME_MODEL:
    resume_model(G, "G", "models_save", LAST_EPOCH)
    resume_model(D1, "D1", "models_save", LAST_EPOCH)
    resume_model(D2, "D2", "models_save", LAST_EPOCH)

optimizer_G = torch.optim.Adam(G.parameters(), lr = LEARNING_RATE)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr = LEARNING_RATE) 
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr = LEARNING_RATE) 

L1 = torch.nn.L1Loss().to(device)

mse = torch.nn.MSELoss().to(device)

cross_entropy = torch.nn.CrossEntropyLoss().to(device)

tb = SummaryWriter(log_dir = '12M_runs')  
epoch = LAST_EPOCH
while(True):
    print("## EPOCH", epoch,"##")
    for step, batch in enumerate(train_dataloader):
        for k in  batch:
            batch[k] =  torch.autograd.Variable( batch[k].to(device) , requires_grad = False) 
        # adversarial loss
        if (step % 500 == 0):
            print(step)
            
        img_40_fake, img_80_fake, img_160_fake = G(batch['img'], batch['landmarks_real'], batch['landmarks_wanted'])

        set_requires_grad( D1 , True )
        #CLASSIC:
        #D1_real = D1(batch['img160_wanted'], batch['img'])
        #L_D1_real = torch.mean(torch.nn.BCEWithLogitsLoss()(D1_real, torch.ones_like(D1_real) * 0.9))
        #D1_fake = D1(img_160_fake.detach(), batch['img'])
        #L_D1_fake = torch.mean(torch.nn.BCEWithLogitsLoss()(D1_fake, torch.zeros_like(D1_fake)))
        #L_D1 = L_D1_real + L_D1_fake
        #L_D1.backward()
        # GRADIENT PENALTY
        D1_real_input = torch.cat([batch['img160_wanted'].data, batch['img'].data], dim = 1)
        D1_fake_input = torch.cat([img_160_fake.detach().data, batch['img'].data], dim = 1)
        D1_loss_no_gp = - torch.mean(D1(torch.cat([batch['img160_wanted'], batch['img']], dim=1))) + torch.mean(D1(torch.cat([img_160_fake.detach(), batch['img']], dim =1)))  
        #compute the gradient penalty
        alpha_1 = torch.rand( D1_real_input.shape[0] , 1 , 1 , 1 ).expand_as(D1_real_input).pin_memory().cuda()
        interpolated_x_1 = torch.autograd.Variable( alpha_1 * D1_fake_input   + (1.0 - alpha_1) * D1_real_input , requires_grad = True) 
        out_1 = D1(interpolated_x_1)
        dxdD_1 = torch.autograd.grad( outputs = out_1 , inputs = interpolated_x_1 , grad_outputs = torch.ones(out_1.size()).cuda() , retain_graph = True , create_graph = True , only_inputs = True  )[0].view(out_1.shape[0],-1)
        gp_loss_1 = torch.mean( ( torch.norm( dxdD_1 , p = 2 ) - 1 )**2 )
        L_D1 = D1_loss_no_gp + 10 * gp_loss_1
        optimizer_D1.zero_grad()
        L_D1.backward()
        optimizer_D1.step()
        set_requires_grad( D1 , False )
        
        
        set_requires_grad( D2 , True )
        # CLASSIC:
        #D2_real = D2(batch['img160_wanted'], batch['landmarks_wanted'])
        #L_D2_real = torch.mean(torch.nn.BCEWithLogitsLoss()(D2_real, torch.ones_like(D2_real) * 0.9))
        #D2_fake = D2(img_160_fake.detach(), batch['landmarks_wanted'])
        #L_D2_fake = torch.mean(torch.nn.BCEWithLogitsLoss()(D2_fake, torch.zeros_like(D2_fake)))
        #L_D2 = L_D2_real + L_D2_fake
        #L_D2.backward()
        # GRADIENT PENALTY
        D2_real_input = torch.cat([batch['img160_wanted'].data, batch['landmarks_wanted'].data], dim = 1)
        D2_fake_input = torch.cat([img_160_fake.detach().data, batch['landmarks_wanted'].data], dim = 1)
        D2_loss_no_gp = - torch.mean(D2(torch.cat([batch['img160_wanted'], batch['landmarks_wanted']], dim=1))) + torch.mean(D2(torch.cat([img_160_fake.detach(), batch['landmarks_wanted']], dim = 1)))  
        #compute the gradient penalty
        alpha_2 = torch.rand( D2_real_input.shape[0] , 1 , 1 , 1 ).expand_as(D2_real_input).pin_memory().cuda()
        interpolated_x_2 = torch.autograd.Variable( alpha_2 * D2_fake_input   + (1.0 - alpha_2) * D2_real_input , requires_grad = True) 
        out_2 = D2(interpolated_x_2)
        dxdD_2 = torch.autograd.grad( outputs = out_2 , inputs = interpolated_x_2 , grad_outputs = torch.ones(out_2.size()).cuda() , retain_graph = True , create_graph = True , only_inputs = True  )[0].view(out_2.shape[0],-1)
        gp_loss_2 = torch.mean( ( torch.norm( dxdD_2 , p = 2 ) - 1 )**2 )
        L_D2 = D2_loss_no_gp + 10 * gp_loss_2
        optimizer_D2.zero_grad()
        L_D2.backward()
        optimizer_D2.step()
        set_requires_grad( D2 , False )

        #G_D1_fake = D1(img_160_fake, batch['img'])
        #G_D2_fake = D2(img_160_fake, batch['landmarks_wanted'])
        
        #adv_D1_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(G_D1_fake, torch.ones_like(G_D1_fake) * 0.9))
        #adv_D2_loss = torch.mean(torch.nn.BCEWithLogitsLoss()(G_D2_fake, torch.ones_like(G_D2_fake) * 0.9))

        adv_D1_loss = -torch.mean(torch.cat([img_160_fake, batch['img']], dim = 1))  
        adv_D2_loss = -torch.mean(torch.cat([img_160_fake, batch['landmarks_wanted']], dim = 1))
        
        pixelwise_160_loss = L1(img_160_fake, batch['img160_wanted'])
        pixelwise_80_loss = L1(img_80_fake, batch['img80_wanted'])
        pixelwise_40_loss = L1(img_40_fake, batch['img40_wanted'])
        features_real = resnet(batch['img160_wanted'])
        features_fake = resnet(img_160_fake)
        

        identity_loss = mse(features_real.detach(), features_fake)
        #identity_loss80 = mse(features_real80.detach(), features_fake80)
        
        total_variation = torch.mean( torch.abs(  img_160_fake[:,:,:-1,:] - img_160_fake[:,:,1:,:] ) )  + torch.mean(  torch.abs( img_160_fake[:,:,:,:-1] - img_160_fake[:,:,:,1:] ) )
        #total_variation80 = torch.mean( torch.abs(  img_80_fake[:,:,:-1,:] - img_80_fake[:,:,1:,:] ) )  + torch.mean(  torch.abs( img_80_fake[:,:,:,:-1] - img_80_fake[:,:,:,1:] ) )  
        
        L_final = 10 * pixelwise_160_loss + 0.1 * pixelwise_80_loss + (1e-4)*pixelwise_40_loss + 0.01 * (adv_D1_loss + adv_D2_loss) + 0.02 * (identity_loss) + 1e-4 *(total_variation)

        optimizer_G.zero_grad()
        L_final.backward()
        optimizer_G.step()
      
        tb.add_scalar( "D1_loss/Train" , L_D1.data.cpu().numpy() ,  epoch*len(train_dataloader) + step)
        tb.add_scalar( "D2_loss/Train" , L_D2.data.cpu().numpy() ,  epoch*len(train_dataloader) + step)
        tb.add_scalar( "pixelwise_160_los/Train" , pixelwise_160_loss.data.cpu().numpy() ,   epoch*len(train_dataloader) + step)
        tb.add_scalar( "pixelwise_80_los/Train" , pixelwise_80_loss.data.cpu().numpy() ,   epoch*len(train_dataloader) + step)
        tb.add_scalar( "pixelwise_40_los/Train" , pixelwise_40_loss.data.cpu().numpy() ,   epoch*len(train_dataloader) + step)
        tb.add_scalar( "identity_loss/Train" , identity_loss.data.cpu().numpy() ,  epoch*len(train_dataloader) + step)
        tb.add_scalar( "total_variation_loss/Train" , total_variation.data.cpu().numpy() ,  epoch*len(train_dataloader) + step)
        tb.add_scalar( "final_loss/Train" , L_final.data.cpu().numpy() , epoch*len(train_dataloader) + step )
   
            
    if (epoch % 2 == 0):
        save_model(G, "G","D:models_NEW_MARKS_12M",  epoch)
        save_model(D1,"D1",  "D:models_NEW_MARKS_12M", epoch)
        save_model(D2,  "D2", "D:models_NEW_MARKS_12M", epoch)
        save_optimizer(optimizer_G,G, "G", "D:optimizer_NEW_MARKS_12M",epoch)
        save_optimizer(optimizer_D1,D1, "D1",  "D:optimizer_NEW_MARKS_12M",epoch)
        save_optimizer(optimizer_D2,D2, "D2", "D:optimizer_NEW_MARKS_12M",epoch)
    
    print("VALIDATION")
    img_batch = np.zeros((8, 3, 160, 160))
    img_batch_index = 0
    identity_mse = 0
    for step, batch in enumerate(test_dataloader):
        for k in  batch:
            batch[k] =  torch.autograd.Variable( batch[k].to(device) , requires_grad = False)
            
        img_40_fake, img_80_fake, img_160_fake = G(batch['img'], batch['landmarks_real'], batch['landmarks_wanted'])
        features_real = resnet(batch['img160_wanted'])
        features_fake = resnet(img_160_fake)

        identity_mse += mse(features_real.detach(), features_fake.clone().detach())
        
        if step % 14 == 0 :
            print(step)
            img_batch[img_batch_index % 8] = img_160_fake[0].cpu().clone().detach()
            img_batch_index += 1
    tb.add_scalar("identity_mse/Test" , identity_mse.data.cpu().numpy() / len(test_dataloader),  epoch)
    
    if (epoch % 2 == 0):
        tb.add_images("img_160_fake/Validation", img_batch, epoch)
    LAST_EPOCH = epoch
    epoch += 1





