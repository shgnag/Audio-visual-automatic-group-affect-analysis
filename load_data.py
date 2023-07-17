'''
Created by Garima
March 28, 2020
'''

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import h5py
import numpy as np
import os

class DatasetProcessing(data.Dataset):
    def __init__(self, label_path,frame_path, vgg_path, audio_features_path):
        super(DatasetProcessing, self).__init__()

        video_list =[]
        label_list = []
        with open(label_path) as id_file:
                for id_ in id_file:
                        if id_.split(' ')[0] != 'Vid_name':
                                video_list.append(id_.split('\n')[0].split(' ')[0])
                                label_list.append(id_.split('\n')[0].split(' ')[1])
        self.video_list = np.asarray(video_list)
        self.label_list = np.asarray(label_list)

        self.frame_path = frame_path
        self.vgg_path = vgg_path
        hf = h5py.File(audio_features_path, 'r') 
        self.audio_features=hf['audio_features'][:].astype('float32')

        
    def __getitem__(self, index):

        vid_name = self.video_list[index]
        label = self.label_list[index]
        label = label -1
        # print(vid_name, label)
        
        
        # vggface
        feature_path = os.path.join(self.vgg_path,vid_name)+'.h5'
        hf = h5py.File(feature_path,'r')
        average_vgg=hf['frame_average_vgg'][:].astype('float32') # taking min, max and average features
        frame_min=hf['frame_min_vgg'][:].astype('float32')
        frame_max=hf['frame_max_vgg'][:].astype('float32')
        hf.close()
        conc_vggface_16f = np.concatenate((average_vgg,frame_min,frame_max),axis = 1) # concatenated shape num_f, 12288

        if(conc_features.shape[0]<16): # Checking the shape of features
            print('Invalid feature shape')
            return
            
        # full frame
        frame_path = os.path.join(self.frame_path,vid_name)+'.h5'
        hf = h5py.File(frame_path,'r')
        full_frame_16f=hf['chunk_frames'][:].astype('float32')
        full_frame_16f = frames/255 # shape num_frame, h, w, c
        # norm_frames = (2 * frames) -1

        # Audio
        audio_features = self.audio_features[index] # extracted feat are of shape 1313


        return (torch.as_tensor(np.array(full_frame_16f).astype('float')),torch.as_tensor(np.array(conc_vggface_16f).astype('float')),torch.as_tensor(np.array(audio_features).astype('float')), torch.as_tensor(np.array(label).astype('float')))
    def __len__(self):
        return len(video_list)