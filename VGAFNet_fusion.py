#!/usr/bin/env python
# coding: utf-8

'''
Created by Garima
March 28, 2020
'''


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # Debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics import accuracy_score
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
import matplotlib.pyplot as plt
SEED = 0
np.random.seed(SEED) # if numpy is used
torch.random.manual_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
from load_data import DatasetProcessing



# Holistic model

model_holistic = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
num_ftrs = model_holistic.fc.in_features
# print(num_ftrs)
# print(model.fc.out_features)
# for param in model.parameters():
#      param.requires_grad = False
# count num of layers

count = 0
for child in model_holistic.children():
    count+=1
# print('no of resnet layers',count)

output_classes = 3
count = 0
for child in model_holistic.children():
    count+=1
    if count < 6:
        for param in child.parameters():
            param.requires_grad = False
            
# model.fc = nn.Linear(num_ftrs, output_classes)
model_holistic = nn.Sequential(*list(model_holistic.children())[:-1])
# set_parameter_requires_grad(model_holistic,True)

model_holistic = model_holistic.to(device)
print(summary(model_holistic,(3,32,112,112)))




# Face level model


class MyLSTM(nn.Module):
    def __init__(self,input_dim,h1,h2):
        super(MyLSTM,self).__init__()
        self.input_dim = input_dim
        self.h1 = h1
        self.h2 = h2

        #LSTM takes, input_dim, hidden_dim and num_layers incase of stacked LSTMs
        self.LSTM_1 = nn.LSTM(input_size = input_dim, hidden_size = h1, num_layers = 1, bidirectional = False)
        self.LSTM_2 = nn.LSTM(input_size  = h1, hidden_size  = h2, num_layers = 1, bidirectional = False)
        self.fc = nn.Linear(h2*32,3)
        
    #Input must be 3 dimensional (seq_len, batch, input_dim). 
    #hc is a tuple of hidden and cell state vector. Each of them have shape (1,1,hidden_dim)
    
    def forward(self,inp):
        batch_size = inp.shape[0]
        inp = inp.view(16,batch_size,12288)
        h0 = torch.zeros(h1*batch_size).view(1,batch_size,h1)
        nn.init.xavier_uniform(h0, gain=nn.init.calculate_gain('relu'))
        c0 = torch.zeros(h1*batch_size).view(1,batch_size,h1)
        nn.init.xavier_uniform(c0, gain=nn.init.calculate_gain('relu'))
        h0 = h0.cuda()
        c0 = c0.cuda()
        #this gives outut for each input in the sequence and also (hidden and cell state vector)
        #Dimensions of output vector is (seq_len,batch,hidden_dim)
        
        output,hc1= self.LSTM_1(inp)
        output,_= self.LSTM_2(output)
        output = output.view(batch_size,-1)
        return output
    
input_dim = 12288
h1 = 256
h2 = 512
h3 = 1024
model_vgg = MyLSTM(input_dim,h1,h2)



# print(model_vgg)
# model_vgg.eval()

model_vgg = model_vgg.to(device)


class audio_dnn(nn.Module):
    def __init__(self):
        super(audio_dnn, self).__init__()
        self.fc1 = nn.Linear(1313, 128) 
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,512)
        self.fc4 = nn.Linear(512,1024)
        self.fc5 = nn.Linear(1024,2048)
        self.fc6 = nn.Linear(2048, 3)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(2048)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc5(x)))
        return x
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)


model_audio = audio_dnn()
model_audio.apply(weight_init)

# print(model_audio)


model_audio = model_audio.to(device)

print(summary(model_audio,(1313,)))



class fusion(nn.Module):
    def __init__(self):
        super(fusion, self).__init__()
        self.bn = nn.BatchNorm1d(512)
        self.fc_vgg = nn.Linear(8192,512)
        self.fc_last = nn.Linear(1536,3)#3072
        self.fc_ad = nn.Linear(2048,512)
        
    def forward(self, x1, x2, x3):
        print(x1.shape, x2.shape, x3.shape)
        x1 = x1.view(-1,512)
        x2 = F.relu(self.bn(self.fc_vgg(x2)))
        x3 = F.relu(self.bn(self.fc_ad(x3)))
        x = torch.cat((x1, x2,x3), 1)
        x = self.fc_last(x)
        return x


model = fusion()
model.apply(weight_init)
model = model.to(device)
params = list(model_holistic.parameters()) +list(model_vgg.parameters()) + list(model_audio.parameters()) + list(model.parameters())



def train(num_epochs,model_name):

    valid_acc_max = 0.0
    valid_loss_min = np.Inf # track change in validation loss
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        train_y_true = []
        train_y_pred = []
        val_y_true = []
        val_y_pred = []
        ###################
        # train the model #
        ###################
        model.train()
        for data_1,data_2,data_3, target in train_loader:

            # move tensors to GPU if CUDA is available
            data_1,data_2,data_3, target = Variable(data_1),Variable(data_2),Variable(data_3), Variable(target)
            target = torch.tensor(target, dtype=torch.long)
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            data_3 = data_3.to(device)
            target = target.to(device)
            data_1 = data_1.permute(0,4,1,2,3)
           # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward pass: compute predicted outputs by passing inputs to the model
            output_audio = model_audio(data_3.float())
            output_vggface = model_vgg(data_2.float())
            output_frame = model_holistic(data_1.float())
            output = model(output_frame,output_vggface,output_audio)

            loss = cce_loss(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data_1.size(0) #loss.data[0]*images.size(0)
            (max_vals, arg_maxs) = torch.max(output, dim=1)
            train_y_pred.extend(arg_maxs.cpu().detach().numpy())
            train_y_true.extend(target.cpu().detach().numpy())

        ######################    
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for data_1,data_2,data_3, target in val_loader:
                data_1,data_2,data_3, target = Variable(data_1),Variable(data_2),Variable(data_3), Variable(target)
                target = torch.tensor(target, dtype=torch.long)
                data_1 = data_1.to(device)
                data_2 = data_2.to(device)
                data_3 = data_3.to(device)
                target = target.to(device)
                data_1 = data_1.permute(0,4,1,2,3)

                output_audio = model_audio(data_3.float())
                output_vggface = model_vgg(data_2.float())
                output_frame = model_holistic(data_1.float())
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(output_frame,output_vggface,output_audio)
                # calculate the batch loss
                loss = cce_loss(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data_1.size(0)
                (max_vals, arg_maxs) = torch.max(output, dim=1)
                val_y_pred.extend(arg_maxs.cpu().detach().numpy())
                val_y_true.extend(target.cpu().detach().numpy())

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(val_loader.sampler)
        train_acc = accuracy_score(np.asarray(train_y_true), np.asarray(train_y_pred))
        val_acc = accuracy_score(np.asarray(val_y_true), np.asarray(val_y_pred))
        
        # print training/validation statistics 
        print('Epoch: {} \tTrain Loss: {:.6f} \tVal Loss: {:.6f} \tTrain acc:{:.5f} \tVal acc:{:.5f} \tTime:{:.2f}'.format(
            epoch, train_loss, valid_loss, train_acc, val_acc, time.time() - epoch_start_time))

        if valid_acc_max <= val_acc:
            print('Validation acc increased ({:.5f} --> {:.5f}).  Saving model ...'.format(
            valid_acc_max, val_acc))
            torch.save(model, model_name) #model.state_dict()
            valid_acc_max = val_acc




if __name__ == '__main__':

    train_label_file_path = 'data/train_labels.txt'
    val_label_file_path = 'data/val_labels.txt'
    
    train_vgg_path = 'data/vggface/train'
    val_vgg_path = 'data/vggface/val'
    
    train_frame_path = 'data/sampled_frames/train' # taking from sampled frames to save memory
    val_frame_path = 'data/sampled_frames/val'
 
    train_audio_features = 'data/audio_features_train.h5'
    val_audio_features = 'data/audio_features_val.h5'
    
    
    batch_size = 2
    num_epochs = 50
    model_name = 'model_holistic_audio_face.pt'
    
    # specify loss function
    cce_loss = nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(params, lr=0.001, weight_decay = 1e-9, momentum = 0.9)
    
#     dset_test = DatasetProcessing(test_label_file_path,test_frame_path,test_vgg_path,test_audio_features)
    dset_val = DatasetProcessing(val_label_file_path,val_frame_path,val_vgg_path,val_audio_features)
    dset_train = DatasetProcessing(train_label_file_path,train_frame_path,train_vgg_path,train_audio_features)

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16)

    val_loader = DataLoader(dset_val,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)

    print('Starting training ----')
    train(num_epochs,model_name)



