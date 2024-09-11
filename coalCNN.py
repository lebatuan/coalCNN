import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
train = pd.read_excel("train.xlsx")
test = pd.read_excel("test.xlsx")
train = train.values
test = test.values

data_train = np.zeros(shape=(train.shape[0], 1, train.shape[1]-1))
data_train_label = np.zeros(shape=(train.shape[0],1))
for i in range(train.shape[0]):
    data_train[i,:,:] = train[i,0:train.shape[1]-1]
    data_train_label[i,:] = train[i,train.shape[1]-1]

data_test = np.zeros(shape=(test.shape[0], 1, test.shape[1]-1))
data_test_label = np.zeros(shape=(test.shape[0],1))
for i in range(test.shape[0]):
    data_test[i, :, :] = test[i,0:test.shape[1]-1]
    data_test_label[i,:] = test[i,test.shape[1]-1]


#print(data_train)


data_train = torch.FloatTensor(data_train)
data_train_label = torch.LongTensor(data_train_label)
data_test = torch.FloatTensor(data_test)
data_test_label = torch.LongTensor(data_test_label)

train_set = TensorDataset(data_train, data_train_label)
test_set = TensorDataset(data_test, data_test_label)

train_dataloader = DataLoader(train_set, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=10, shuffle=True)

#for k in train_dataloader:
#    imgs, targets = k
#    print(imgs)
#    print(imgs.shape)
#    print(targets)
#    print(targets.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 25, 3),   #
            nn.Conv1d(25, 25, 3),  #
            nn.Conv1d(25, 25, 3),  #
            nn.Conv1d(25, 25,  3),  #
            nn.Conv1d(25, 25,  3),  # [10,25,7284]-------->[10,25,7185]
            nn.Conv1d(25, 25,  3),  # [10,25,7185]-------->[10,25,7086]
            nn.BatchNorm1d(25),  # [10,25,7086]-------->[10,25,7086]
            nn.ReLU(),  # [10,25,7086]-------->[10,25,7086]

            nn.MaxPool1d(2, stride=2),  # [10,25,7086]-------->[10,25,3543]

            nn.Conv1d(25, 25,  3),  # [10,25,3543]-------->[10,25,3444]
            nn.Conv1d(25, 25,  3),  # [10,25,3444]-------->[10,25,3345]
            nn.Conv1d(25, 25,  3),  # [10,25,3345]-------->[10,25,3246]
            nn.BatchNorm1d(25),  # [10,25,3246]-------->[10,25,3246]
            nn.ReLU(),  # [10,25,3246]-------->[10,25,3246]

            nn.MaxPool1d(2, stride=2),  # [10,25,3246]-------->[10,25,1623]

            nn.Conv1d(25, 50,  3),  # [10,25,1623]-------->[10,50,1524]
            nn.Conv1d(50, 50,  3),  # [10,50,1524]-------->[10,50,1425]
            nn.Conv1d(50, 50,  3),  # [10,50,1425]-------->[10,50,1326]
            nn.BatchNorm1d(50),  # [10,50,1326]-------->[10,50,1326]
            nn.ReLU(),  # [10,50,1326]-------->[10,50,1326]

            nn.MaxPool1d(2, stride=2),  # [10,50,1326]-------->[10,50, 663]

            nn.Conv1d(50,  100,  3),  # [10,50, 663]-------->[10,100,564]
            nn.Conv1d( 100,  100,  3),  # [10,100,564]-------->[10,100,465]
            nn.Conv1d( 100,  100,  3),  # [10,100,465]-------->[10,100,366]
            nn.BatchNorm1d( 100),  # [10,100,366]-------->[10,100,366]
            nn.ReLU(),  # [10,100,366]-------->[10,100,366]

            nn.MaxPool1d(2, stride=2),  # [10,100,366]-------->[10,100,183]

            nn.Conv1d( 100,  100, 5),  # [10,100,183]-------->[10,100,120]

            nn.MaxPool1d(10, stride=5),  # [10,100,120]-------->[10,100, 12]

            nn.Conv1d( 100, 4, 5),  # [10,100, 12]-------->[10,  4,  8]
            nn.BatchNorm1d(4),  # [10,  4,  8]-------->[10,  4,  8]
            nn.ReLU(),  # [10,  4,  8]-------->[10,  4,  8]

            nn.Flatten(),
            nn.Linear(20, 16),
            nn.Linear(16, 4),   # output[10,5]
        )

    def forward(self, x):
        x = self.model(x)
        return x



net = Net()
if torch.cuda.is_available():
    net = net.cuda()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


total_train_step = 0

total_test_step = 0

epoch = 100

for i in range(epoch):
    print("-----Iteration {} -----".format(i + 1))
    
    net.train()
    for k in train_dataloader:
       
        imgs, targets = k
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = net(imgs)
       
        loss = loss_fn(outputs, targets.squeeze(1).long())
      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("Iteration {} loss {} ".format(total_train_step, loss.item()))
    
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for k in test_dataloader:
            imgs, targets = k
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fn(outputs, targets.squeeze(1).long())
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets.squeeze(1).long()).sum()
            total_accuracy = total_accuracy + accuracy
    total_test_step = total_test_step + 1
   
    print("acc is {} ".format(total_accuracy / 119))
