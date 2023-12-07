import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import time

device = torch.device('cuda') #setting the m1 device

train_data = datasets.MNIST(root = './python-mnist/bin/data',
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
#creating a pytorch data loader 
#setting parameters
batch_size = 60000
learning_rate = 0.0001
epoch_num = 10

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size = batch_size, shuffle = True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        #input and output layer
        self.fc1 = nn.Linear(3136, 1000)    # 7 * 7 * 64 = 3136
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
        

model = CNN().to(device)
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

model.train()

i = 1

for epoch in range(epoch_num):
    start = time.time()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("Train Step : {}\tLoss : {:3f}".format(i, loss.item()))
        i += 1
    end = time.time()
    print(end - start)