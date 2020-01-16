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
        test(prnt=False, it=it, epoch=epoch)

def test(prnt=False, it=-1, epoch=-1):
    model.eval()
    test_loss = 0
    correct = 0
    yer = 0
    per = 0
    rer = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)     
            test_loss += loss*3*output.shape[0]
            for i in range(output.shape[0]):
                    yer += abs(output[i][0]-target[i][0])
                    per += abs(output[i][1]-target[i][1])
                    rer += abs(output[i][2]-target[i][2])
    test_loss /= (len(test_loader.dataset)*3)
    print ('Test samples: %d Test Loss: %.5f'%(len(test_loader.dataset), test_loss.item()))

    #Convert to Degrees
    er1 = ((yer.item())/len(test_loader.dataset))*180
    er2 = ((per.item())/len(test_loader.dataset))*180
    er3 = ((rer.item())/len(test_loader.dataset))*180

    #If new error is lesser, update the error variables
    if er1+er2+er3 < fer[0] + fer[1] + fer[2]:
        print(epoch, "improved", (er1+er2+er3)/3)
        fer[0] = er1
        fer[1] = er2
        fer[2] = er3
    
#Checking for Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device,use_cuda)

#Set your batch size and number of epochs here
#Default Settings: Batch_size = 128 , Num_Epochs = 1020
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
BATCH_SIZE = 128
NB_EPOCHS = 1020
fer = 1000*np.ones((3))
path_and_ground_truth_file = 'biwiGT'

#Creating the training and testing sets
als = []
for num in range(1, 25):
    als.append(num)
als = set(als) 
selected_test_set = [12, 16, 17]    
train_set = als-set(selected_test_set)
train_set = list(train_set)
print(train_set, selected_test_set)
print("\n==============================\n")

#Training dataset
train_dataset = biwiDataset(path_and_ground_truth_file, 1)
train_dataset.select_sets(sets=train_set)

#Testing dataset
test_dataset = biwiDataset(path_and_ground_truth_file, 0)
test_dataset.select_sets(sets=selected_test_set)

#Preparing both the datasets to be fed into the CNN
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = simpNet().to(device)
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

#Using Adam's optimiser and setting the learning rate ti 10e-5
optimizer = optim.Adam(model.parameters(), lr = 0.000001)
loss_func = torch.nn.MSELoss()

#Training
train(NB_EPOCHS, it=1) 

#Testing, comment the below line if you don't want to test the model
test(prnt=False, it=1)  

#Average Error Readings in degrees
print('MAE: Yaw %.5f, Pitch %.5f, Roll %.5f, Avg %.5f'%(fer[0], fer[1], fer[2], (fer[0]+fer[1]+fer[2])/3))

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
