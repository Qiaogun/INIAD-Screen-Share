import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import the libraries for classification modelling
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error

from data_processing.datasets import *
from train_model import train_model
from test_model import test_model
from plot_model_stats import plot_model_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNLSTMNet(nn.Module):
    def __init__(self, cnn_model, output_dim, hidden_dim,num_classes=10,seq_len=20 ,batch_size=1, num_lstm_layers = 1, bidirectional = False, device = 'cpu', freeze_layers=True, dropout=0, title="default"):
        super(CNNLSTMNet, self).__init__()
        # CNN
        self.device = device
        self.title = title # Model Title
        self.cnn_model = cnn_model # Torchvision CNN Model
        
        # Optionally Freeze CNN Layers
        if freeze_layers:
            for idxc, child in enumerate(self.cnn_model.children()):
                for param in child.parameters():
                    param.requires_grad = False
            self.cnn_model.fc.requires_grad = True
            
        # RNN
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim # LSTM Hidden Dimension Size
        self.num_lstm_layers = num_lstm_layers 
        self.bidirectional = bidirectional # Sets LSTM to Uni or Bidirectional
        self.bidirectional_mult = 2 if self.bidirectional else 1 # Used for LSTM Weight Shape
        self.lstm = nn.LSTM(output_dim, hidden_dim, self.num_lstm_layers, bidirectional=self.bidirectional, dropout=dropout)
        self.hidden2class = nn.Linear(hidden_dim*self.bidirectional_mult, num_classes) # Fully Connected Output Layer
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_lstm_layers*self.bidirectional_mult, self.batch_size, self.hidden_dim).to(device),
                    torch.zeros(self.num_lstm_layers*self.bidirectional_mult, self.batch_size, self.hidden_dim).to(device))


    def forward(self, x):
        x = x.view(self.seq_len*self.batch_size,x.shape[-3],x.shape[-2],-1)
        out = self.cnn_model(x)
        seq = out.view(self.batch_size, self.seq_len, -1).transpose_(0,1)
        self.hidden = self.init_hidden()
        # LSTM input shape = (seq_len, batch, input_size)
        out, self.hidden = self.lstm(seq.view(len(seq), self.batch_size, -1), self.hidden)
        #LSTM output shape = (seq_len, batch, hidden_dim * bidirectional)
        out = self.hidden2class(out[-1])
        return out


# root_dir='./'
# train_transform = transforms.Compose(
#     [
#         transforms.ToPILImage(),
#         transforms.Resize(256),  # 1. Resize smallest side to 256.
#         transforms.CenterCrop(224), # 2. Crop the center 224x224 pixels.
#         transforms.ToTensor(),
#         transforms.Normalize(mean = [0.485, 0.456, 0.406],  # Normalize. This is necessary when using torchnet pretrained models.
#                           std = [0.229, 0.224, 0.225])
#     ])

# trainset = HMDB(30, root_dir=root_dir, transforms=train_transform) #Use ending 20 frames from each clip
# print("Train size:",len(trainset))


def display_sequence(trainset):
    print('Labels:', trainset.labels)
    
    # Sample the dataset
    rand_int = np.random.randint(0,len(trainset))
    sample_video, label = trainset[rand_int]
    video_label = trainset.data_file_labels[rand_int]
    num_frame = 6 # Number of frames to display
    
    # Display Frames
    frames = np.asarray(transforms.ToPILImage()(sample_video[0]))
    print('Data Shape:', sample_video.shape)
    for i in range(1, trainset.seq_len,int(trainset.seq_len/num_frame)):
        frame = sample_video[i]
        for t, m, s in zip(frame, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        frame = np.asarray(transforms.ToPILImage()(frame))
        frames = np.concatenate((frames, frame), axis=1)
        print(i, "frame size:", frame.shape, 'label:', video_label)
    print('Frame sequence')
    print(frames.shape)
    print('Visualize the data where the first image is normalized, and the rest are not.')
    plt.figure(figsize=(50, 10))
    plt.grid(False)
    plt.imshow(frames)


display_sequence(trainset)

num_img_features = 1024 # CNN output dimensions
num_epochs = 10
sequence_len = trainset.seq_len # LSTM sequence length
batch_size=3
hidden_dim = 128 # LSTM hidden dimension size
lstm_dropout = .1
lstm_depth = 1
freeze = False # True = Freeze entire CNN, False = Don't freeze any layers
pretrain = True # Use Imagenet Pretraining with CNN
lstm_depth_title = 'no_freeze_layers_num_lstm_layers_'+str(lstm_depth)
num_classes = len(trainset.labels)
loss_fn = nn.CrossEntropyLoss()


title =\
'-dataset-'+str(trainset.title)+\
'-frozen-'+str(freeze)+\
'-num_img_features-'+ str(num_img_features) +\
'-num_epochs-'+str(num_epochs)+\
'-sequence_len-'+str(sequence_len)+\
'-batch_size-'+str(batch_size)+\
'-lstm_dim-'+str(hidden_dim)+\
'-lstm_depth-'+str(lstm_depth)+\
'-pretrain-'+str(pretrain)+\
'CNN-resnet18'

cnn_model = models.resnet18(pretrained=pretrain) #Choose different CNN if desired
num_ftrs = cnn_model.fc.in_features
cnn_model.fc = nn.Linear(num_ftrs, num_img_features) # Change CNN output layer to desired dimension
model = CNNLSTMNet(cnn_model, num_img_features, hidden_dim, num_classes, sequence_len, batch_size, num_lstm_layers=lstm_depth, bidirectional=False, device=device, freeze_layers=freeze, dropout=lstm_dropout)
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
model_save_path= 'saved_models/'+title+'.pth'
# Use Below to load train history and train over a saved model
# model.load_state_dict(torch.load(model_save_path))
print(title)

train_model(model, loss_fn, batch_size, trainset, optimizer, title, device, num_epochs=num_epochs)