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
from sklearn.model_selection import train_test_split


class cnn_classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(1,32,1), stride=(1,32,1))
    self.pool1 = nn.AvgPool3d(kernel_size=(2,1,1), padding=0)
    
    self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1,1,3),stride=(1,1,3),padding=(0,0,1))
    self.pool2 = nn.MaxPool3d(kernel_size=(1,1,2), padding=0)
    
    self.fc_layer = nn.Linear(64*16*2, 2)
    
    self.dropout_layer = nn.Dropout(p=0.5)

  def forward(self, xb):
    h1 = self.conv1(xb)
    #h1 = self.dropout_layer(h1)
    h1 = self.pool1(h1)
    h1 = F.relu(h1)
    

    h2 = self.conv2(h1)
    #h2 = self.dropout_layer(h2)
    h2 = self.pool2(h2)
    h2 = F.relu(h2) 
    
    # flatten the output from conv layers before feeind it to FC layer
    h2 = h2.view(-1, 64*16*2)
    out = self.fc_layer(h2)    
    return out

def train_model(model, x_train, y_train, x_test, y_test, epochs=30 , batch_size=64, lr=0.0001, weight_decay=0):
    # data
  train_dataset = TensorDataset(x_train, y_train)
  train_data_loader = DataLoader(train_dataset, batch_size=batch_size)

  # loss function
  loss_func = F.cross_entropy

  # optimizer
  optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
  #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

  # figure
  train_a = list([])
  test_a = list([])

  # training loop
  for epoch in range(epochs):
    tic = time.time()
    for xb, yb in train_data_loader:    
      pred = model(xb)
      loss = loss_func(pred, yb.long())

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    toc = time.time()

    y_pred = model(x_test)
    y_pred_train = model(x_train)
    acc = accuracy_score(torch.argmax(y_pred, dim=1).detach().numpy(), y_test)
    acc_train = accuracy_score(torch.argmax(y_pred_train, dim=1).detach().numpy(), y_train)

    train_a.append(acc_train)
    test_a.append(acc)
    print('Loss at epoch %d : %f, train_acc: %f, test_acc: %f, running time: %d'% (epoch, loss, acc_train, acc, toc-tic))

  # draw an accuray figure
  plt.plot(train_a,'y.-.')
  plt.plot(test_a,'.-.')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')



# load data
dfs = []
for i in range(1,33):
  for j in range(1,41):
    filename = r'C:\Users\75196\Documents\GitHub\SchoolCourse\PatternRec\EmotionRec\CWT\File_10frame_exscale\participant%dvideo%d.txt'%(i,j)
    cols = [i for i in range(10)]
    df = pd.read_csv(filename, header = None,usecols = cols, delimiter=',')   
    dfs.append(df.values)
    #print('participant%dvideo%d.txt'%(i,j))

dfs = np.array(dfs)
print('dataLoaded:')
print(dfs.shape)

# load label
cols = ['valence', 'arousal', 'dominance', 'liking']
label_df = pd.read_csv(r'C:\Users\75196\Documents\GitHub\SchoolCourse\PatternRec\EmotionRec\CWT\label.txt',
    usecols = [i for i in range(4)], header=None, delimiter=',' )
print(label_df.shape)
label_df.columns = cols
label_df[label_df<5] = 0
label_df[label_df>=5] = 1

# valence
label = label_df['valence'].values
print(label.size)

# divive train & test
x_train, x_test, y_train, y_test = train_test_split(
  dfs, label, test_size=0.2, random_state=1)

# turn into Tensors
#x_train, y_train, x_test, y_test = map(
#    torch.from_numpy, (x_train, y_train, x_test, y_test)
#)
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

n = x_train.shape[0]
print(n)

cnn_model = cnn_classifier()
train_model(cnn_model, x_train.view(-1, 1, 32, 32, 10), y_train, x_test.view(-1, 1, 32, 32, 10), y_test)