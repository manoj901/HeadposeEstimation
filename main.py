from __future__ import print_function, division
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from biwiDataLoader import biwiDataset
from simpleNet import simpNet
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random      
import time

def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def train(epochs, it):
    model.train()
    for epoch in range(0,epochs):
        train_loss = 0
        t = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   
            output = model(data)
            loss = loss_func(output, target)  
            train_loss += loss*output.shape[0]*3   
            loss.backward() 
            optimizer.step()

        #MSE loss has a factor of 1/3
        train_loss = train_loss/(len(train_loader.dataset)*3)   
        elapsed = time.time() - t
        print (' -%d-  Epoch [%d/%d]'%(elapsed, epoch+1, NB_EPOCHS))
        print ('Training samples: %d Train Loss: %.5f'%(len(train_loader.dataset), train_loss.item()))
    
#Checking for Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device,use_cuda)

#Set your batch size and number of epochs here
#Default Settings: Batch_size = 128 , Num_Epochs = 1020
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
BATCH_SIZE = 128
NB_EPOCHS = 1020
path_and_ground_truth_file = 'biwiGT'

#Creating the training set
train_set = []
for num in range(1, 25):
    train_set.append(num)

#Training dataset
train_dataset = biwiDataset(path_and_ground_truth_file, 1)
train_dataset.select_sets(sets=train_set)

#Preparing the dataset to be fed into the CNN
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = simpNet().to(device)
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

#Using Adam's optimiser and setting the learning rate to 10e-5
optimizer = optim.Adam(model.parameters(), lr = 0.000001)
loss_func = torch.nn.MSELoss()

#Training
train(NB_EPOCHS, it=1) 

#Add your heatmap's address to the own_set list to find out it's yaw, pitch and roll
#Sample heatmap addresses have been added to the list to give an idea. You can change this list as you wish
own_set = ['./Extra/cena_out_000000000000_pose_heatmaps.png','./Extra/man_out_000000000000_pose_heatmaps.png','./Extra/mes_out_000000000000_pose_heatmaps.png','./Extra/vir_out_000000000000_pose_heatmaps.png']

for xi in own_set:
	img = Image.open(xi)

    #Extracting the 5 keypoint heatmaps from heatmap
	i=0
	hmroi = (96*i,0,96*(i+1),96)
	data = [] 
	tmp = img.crop(hmroi)
	data.append(tmp)	
	for i in range(1, 5):	
	    hmroi = (96*(i+13),0,96*(i+14),96)
	    tmp = img.crop(hmroi)
	    data.append(tmp)
	margin = 12
	hm = data
	x = random.randint(0,margin)
	y = random.randint(0,margin)
	xd = 96-random.randint(0,margin)    
	yd = 96-random.randint(0,margin)   
	nhm = [] 
	for i in range(0,5):
	    nhm.append(hm[i].crop((x,y,xd,yd)))
	fhm = []       
	for i in range(0,5):
	    fhm.append(np.array(nhm[i].resize((96, 96), Image.ANTIALIAS)))

	hmc = np.zeros((5,96, 96))
	for i in range(0,5):	
	    hmc[i,:,:] = fhm[i]

	img = hmc
	img = img.astype('float32')
	img /= 255
	img = [img]
	img = np.asarray(img)
	data = torch.from_numpy(img)	
	data = data.to(device)
	output = model(data)

    #Print heatmap name
	print(xi)

    #Print Yaw, Pitch and Roll output
    #Note: You have to multipy the angles by 180 and then subtract 90 from it to..
    #..get these angles in degrees
	print(output)
