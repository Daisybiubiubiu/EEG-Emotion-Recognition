import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time

from sklearn.metrics import accuracy_score


class cnn_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, padding=(0,0,1))
        
        self.conv21 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, padding=(0,0,1))
        
        self.conv31 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, padding=0)
        

        self.fc_layer = nn.Linear(128*4*4*1, 2)
        
        self.dropout_layer = nn.Dropout(p=0.5)

    def forward(self, xb):
        h1 = self.conv11(xb)
        h1 = self.conv12(h1)
        h1 = self.dropout_layer(h1)
        h1 = self.pool1(h1)
        h1 = F.relu(h1)

        h2 = self.conv21(h1)
        h2 = self.conv22(h2)
        #h2 = self.dropout_layer(h2)
        h2 = self.pool2(h2)
        h2 = F.relu(h2) 

        h3 = self.conv31(h2)
        h3 = self.conv32(h3)
        #h3 = self.dropout_layer(h3)
        h3 = self.pool3(h3)
        h3 = F.relu(h3) 
        
        
        # flatten the output from conv layers before feeind it to FC layer
        flatten = h3.view(-1, 128*4*4*1)
        out = self.fc_layer(flatten)
        #out = self.dropout_layer(out)
        return out


def predict(model, x_test, y_test):
    model.eval()
    y_pred = model(x_test.to(device))
    acc = y_pred.argmax(1).eq(y_test.to(device)).float().mean().cpu().numpy()
    print('High valence: 1 & Low valence: 0')
    print('Label: ' + str(y_test))
    print('CNN Output: ' + str(y_pred))
    print('Label predicted: ' + str(y_pred.argmax(1)))
    print('test acc: %f'%acc)
    

path = r'C:\Users\75196\Documents\GitHub\SchoolCourse\PatternRec\EmotionRec\CWT'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load test data
x_test = torch.load(path+'/x_test.pth', map_location=device)
y_test = torch.load(path+'/y_test.pth', map_location=device)

# select first data to test
x = x_test[0]
y = y_test[0]

# load model & optimizer
model = cnn_classifier()
model.load_state_dict(torch.load(path+'/model_statedict.pth', map_location=device))
#optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

predict(model, x.view(-1, 1, 32, 32, 3), y)
print('done')
