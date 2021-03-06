from __future__ import print_function, division
import os,io
import torch
import numpy as np
import PIL
from PIL import Image
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from skimage.transform import resize

class biwiDataset(Dataset):

    def __init__(self, path_file, trnFlg=0, transform=None):
        """Gather the 5 heatmaps from each heatmap image"""

        f = open(path_file,"r")
        self.alines = f.readlines()
        self.transform = transform
        self.trnFlg = trnFlg
        self.allImgs = []
        self.allgt = []
        self.allsn = []

        self.width = 96
        self.height = 96
        self.heatmaps_cnt = 5

        for idx in range(0,len(self.alines)):
                line = self.alines[idx]
                line = line.split()
                nm = line[0].split("/")[-1]
                seq_no = int(line[0].split("/")[-2]) 
                lbl = np.float32(np.asarray([-float(line[1]), float(line[3]), -float(line[2])]))
                prts = nm.split("_")
                namehm = prts[0] + '_' + prts[1] + '_rgb_c' + line[0].split("/")[-2] + '_heatmaps.png'  
                isPrs = True
                img = Image.open('all_heatmaps/'+namehm)
                i=0
                hmroi = (self.width*i,0,self.width*(i+1),self.height)
                hm = [] 
                tmp = img.crop(hmroi)
                hm.append(tmp)	
                for i in range(1, self.heatmaps_cnt):	
                    hmroi = (self.width*(i+13),0,self.width*(i+14),self.height)
                    tmp = img.crop(hmroi)
                    hm.append(tmp)	
                self.allImgs.append(hm)
                self.allgt.append(lbl)
                self.allsn.append(seq_no)
    
    
    def __getitem__(self, idx):
        """Pre-process the heatmap and angles for the CNN"""

        hm = self.Imgs[idx]
        lbl = self.gt[idx]
        margin = 12
        x = random.randint(0,margin)
        y = random.randint(0,margin)
        xd = self.height-random.randint(0,margin)    
        yd = self.width-random.randint(0,margin)
           
        nhm = [] 
        for i in range(0,self.heatmaps_cnt):
            nhm.append(hm[i].crop((x,y,xd,yd)))
        fhm = []       
        for i in range(0,self.heatmaps_cnt):
            fhm.append(np.array(nhm[i].resize((self.width, self.height), PIL.Image.ANTIALIAS)))
        
        hmc = np.zeros((self.heatmaps_cnt,self.width, self.height))
        for i in range(0,self.heatmaps_cnt):	
            hmc[i,:,:] = fhm[i]
      
        img = hmc
        img = img.astype('float32')
        img /= 255
        img = torch.from_numpy(img)
        lbl += 90
        lbl /= 180
        
        lbl =  torch.from_numpy(lbl)
        if self.transform:
            img = self.transform(img)
        return img, lbl
	
    def __len__(self):
        return len(self.Imgs)

    def to_categorical(self, y, num_classes):
        return np.eye(num_classes, dtype='uint8')[y]

    def select_sets(self, sets=[]):
        """Creation of training and testing sets"""

        self.Imgs = []
        self.gt = []
        self.lines = []
        if len(sets)==0:
            for idx in range(0,len(self.alines)):    
                self.Imgs.append(self.allImgs[idx])
                self.gt.append(self.allgt[idx])   
        else:
            for idx in range(0,len(self.alines)):
                if self.allsn[idx] in sets:
                    self.Imgs.append(self.allImgs[idx])
                    self.gt.append(self.allgt[idx]) 
                    self.lines.append(self.alines[idx])  
 