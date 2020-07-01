import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os
#from tqdm import tnrange, tqdm, tqdm_notebook
from tqdm import tqdm, trange
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.nn import init
dtype = torch.cuda.FloatTensor


def k(distsq,D,dt):
    return torch.exp(-distsq/(4*D*dt))/(4*np.pi*D*dt)
def gradientk(dist,D,dt):
    return dist*torch.exp(-(dist**2).sum(1)[:,None,:,:,:,:]/(4*D*dt))/(8*np.pi*D**2*dt**2)
class GaussianConvolution(Function):
    D = 0.14
    dt = 1
    @staticmethod
    def forward(ctx, w, I):
        ctx.save_for_backward(w,I)
        interval=torch.arange(I.size()[-1]).type(dtype)
        x1 = interval[None,:,None,None,None]
        x2 = interval[None,None,:,None,None]
        y1 = interval[None,None,None,:,None]
        y2 = interval[None,None,None,None,:]
        distsq = (x1-y1-w[:,0,:,:,None,None])**2+(x2-y2-w[:,1,:,:,None,None])**2
        return (I[:,None,None,:,:]*k(distsq,GaussianConvolution.D,GaussianConvolution.dt)).sum(4).sum(3)
    
    @staticmethod
    def backward(ctx, grad_output):
        w,I = ctx.saved_variables
        w=w.data
        I=I.data
        interval=torch.arange(I.size()[-1]).type(dtype)
        x1 = interval[None,:,None,None,None]
        x2 = interval[None,None,:,None,None]
        y1 = interval[None,None,None,:,None]
        y2 = interval[None,None,None,None,:]
        distx = (x1-w[:,0,:,:,None,None]-y1)[:,None,:,:,:,:].repeat(1,1,1,1,1,I.size()[-1])
        disty = (x2-w[:,1,:,:,None,None]-y2)[:,None,:,:,:,:].repeat(1,1,1,1,I.size()[-1],1)
        dist = torch.cat((distx,disty),dim=1)
        grad = Variable((I[:,None,None,None,:,:]*gradientk(dist,GaussianConvolution.D,GaussianConvolution.dt)).sum(5).sum(4), requires_grad=False)
        #I(x) only depends on w(x) and not on w(z) for z != x
        return grad*grad_output[:,None,:,:], None


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.xavier_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LeakyReLU):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
class CDNN(nn.Module):
    def __init__(self):
        super(CDNN, self).__init__()
        self.down_conv1 = nn.Conv2d(4,64,3,stride=1,dilation=16)
        self.down_conv2 = nn.Conv2d(64,128,3,stride=1,dilation=8)
        self.down_conv3 = nn.Conv2d(128,256,3,stride=1,dilation=4)
        self.down_conv4 = nn.Conv2d(256,512,3,stride=1,dilation=2)
        
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        
        self.l_relu3 = nn.LeakyReLU(0.1)
        self.l_relu2 = nn.LeakyReLU(0.1)
        self.l_relu1 = nn.LeakyReLU(0.1)
        
        self.trans_conv4 = nn.ConvTranspose2d(512,386,3,stride=1,dilation=2)
        self.trans_conv3 = nn.ConvTranspose2d(386,194,3,stride=1,dilation=4)
        self.trans_conv2 = nn.ConvTranspose2d(194,98,3,stride=1,dilation=8)
        self.trans_conv1 = nn.ConvTranspose2d(98,2,3,stride=1,dilation=16)
    
    def init_weights(self):
        initialize_weights(self.down_conv1, self.down_conv2, self.down_conv3, self.down_conv4, 
                           self.batchnorm1, self.batchnorm2, self.batchnorm3, 
                           self.trans_conv4, self.trans_conv3, self.trans_conv2, self.trans_conv1
                          )
        
    def forward(self, x):
        interm1 = self.batchnorm1(self.down_conv1(x))
        interm2 = self.batchnorm2(self.down_conv2(interm1))
        interm3 = self.batchnorm3(self.down_conv3(interm2))
        w = self.l_relu3(self.trans_conv4(self.down_conv4(interm3)))
        padding = Variable(torch.zeros(x.size()[0],386-256,8,8).type(dtype),requires_grad=False)
        w += torch.cat((interm3,padding),dim=1)
        w = self.l_relu2(self.trans_conv3(w))
        padding = Variable(torch.zeros(x.size()[0],194-128,16,16).type(dtype),requires_grad=False)
        w += torch.cat((interm2,padding),dim=1)
        w = self.l_relu1(self.trans_conv2(w))
        padding = Variable(torch.zeros(x.size()[0],98-64,32,32).type(dtype),requires_grad=False)
        w += torch.cat((interm1,padding),dim=1)
        w = self.trans_conv1(w)
        return w


def charbonnier(x,epsilon=0,alpha=1/2):
    return (x+epsilon)**(1/alpha)

lam_div=1
lam_magn=-0.03
lam_grad=0.4
def Loss(w,pred,target,epsilon=0,alpha=1/2):
    #without derivatives at the borders
    dxkernel = Variable(torch.cuda.FloatTensor([[[[-1,1]]]]).repeat(1,2,1,1),requires_grad=False)
    dykernel = Variable(torch.cuda.FloatTensor([[[[-1],[1]]]]).repeat(1,2,1,1),requires_grad=False)
    dwdx = nn.functional.conv2d(w,dxkernel)[:,:,:63,:]
    dwdy = nn.functional.conv2d(w,dykernel)[:,:,:,:63]
    div = ((dwdx+dwdy)**2).sum()
    magn = (w[:,0]**2+w[:,1]**2).sum()
    grad = (dwdx**2+dwdy**2).sum()
    #return charbonnier(pred-target,epsilon,alpha).sum()+lam_div*div+lam_magn*magn+lam_grad*grad
    return charbonnier(pred-target).sum()


areanumbers = list(range(1,30))#[17,18,19]
#areanumbers = [17,18,19]
batchsize=17
#batchsize=10
train_test_split = 3292 #2016-01-01
#train_test_split = 100*4
#train_test_split=6
val_amount = train_test_split//5


# In[14]:


def hours_to_datestring(t):
    t=int(t)
    start = datetime.datetime(1950,1,1)
    delta = datetime.timedelta(hours=t)
    return (start+delta).strftime('%Y-%m-%d %H:%M:%S')
# Limit to 2017-12-23 for standardisation
times = np.load('dataset/times.npy')[:365*11]
print(hours_to_datestring(times[0]))
print(hours_to_datestring(times[3292]))


# In[15]:


areatemps=[]
for i in areanumbers:
    print(i)
    nextarea = (np.load(f'dataset/by-area/area{i}.npy')[:365*11])
    # Ignore leap days
    for i in range(365):
        days = nextarea[i::365]
        days -= np.mean(days)
        days /= np.std(days)
    areatemps.append(nextarea)
areatemps = np.array(areatemps)


# In[16]:


def input_indices(end_indices):
    a = end_indices[:,None]
    return np.concatenate((a-4,a-3,a-2,a-1),axis=1)

def new_size(size):
    return torch.Size((size[0]*size[1],))+size[2:]
train_val_beginnings = 4+np.random.permutation(train_test_split-4)
train_inputs = torch.from_numpy(areatemps[:,input_indices(train_val_beginnings[val_amount:])]).clone()
train_inputs = train_inputs.view(new_size(train_inputs.size()))

train_ends = torch.from_numpy(areatemps[:,train_val_beginnings[val_amount:]]).clone()
train_ends = train_ends.view(new_size(train_ends.size()))

train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_inputs,train_ends),
                                         batch_size=batchsize, shuffle=True, pin_memory=False)


val_inputs = torch.from_numpy(areatemps[:,input_indices(train_val_beginnings[:val_amount])]).clone()
val_inputs = val_inputs.view(new_size(val_inputs.size()))

val_ends = torch.from_numpy(areatemps[:,train_val_beginnings[:val_amount]]).clone()
val_ends = val_ends.view(new_size(val_ends.size()))

val_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_inputs,val_ends),batch_size=batchsize,
                                      shuffle=False,pin_memory=False)
"""
test_inputs = torch.from_numpy(areatemps[:,train_test_split+4:])
test_inputs = Variable(test_inputs.view(new_size(test_inputs.size())), requires_grad=False)

test_ends = torch.from_numpy(areatemps[:,input_indices(np.arange(train_test_split+4,areatemps.shape[1]))])
test_ends = Variable(test_ends.view(new_size(test_ends.size())), requires_grad=False)
"""


# In[9]:


class StateSaver:
    def __init__(self, name, every=10, i=0):
        self.every = every
        self.name = name
        self.i = i
    
    def save(self, state_dict, loss=None):
        self.i += 1
        if self.i % self.every == 0:
            num = self.i // self.every
            if not os.path.exists(f'models/{self.name}'):
                os.makedirs(f'models/{self.name}')
            torch.save(state_dict,f'models/{self.name}/{self.i}.state')
            if loss is not None:
                torch.save(loss,f'models/{self.name}/{self.i}.loss')



net = CDNN()
torch.nn.modules.module.Module.cuda(net)
net.init_weights()
#net.load_state_dict(torch.load('models/test1/9.state'))
learning_rate=1e-7
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
warping = GaussianConvolution.apply
saver = StateSaver(f'full-training',every=1)
#losses = list(torch.load('models/test1/9.loss'))
losses = []
epochloss = 0
for epoch in trange(2,desc='epoch',leave=False):
    for data in tqdm(train_data,leave=False, desc='batch'):
        Input, Target = data
        Input = Variable(Input.type(dtype),requires_grad=False)
        Target = Variable(Target.type(dtype),requires_grad=False)
        w = net(Input)
        optimizer.zero_grad()
        pred = warping(w, Input[:,-1])
        loss = Loss(w,pred,Target,epsilon=0.5,alpha=1)
        epochloss += loss.data[0]
        loss.backward()
        optimizer.step()
    losses.append(epochloss)
    saver.save(net.state_dict(), np.array(losses))

