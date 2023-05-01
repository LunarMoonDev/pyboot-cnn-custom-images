# importing libraries
import enum
import os
import time
from random import shuffle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvolutionalNetwork

# constants
DATA = './data/processed'
VERSION = '0.0'

# preping data
transform = transforms.ToTensor()
train_data = datasets.ImageFolder(os.path.join(DATA, 'train'), transform = transform)
test_data = datasets.ImageFolder(os.path.join(DATA, 'test'), transform = transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = True)

# preping the model
torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr = 0.001)

start_time = time.time()
epochs = 3
max_trn_batch = 800
max_tst_batch = 300

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):

        ## limit the number of batches
        ## i think shuffle for the loader is already resolved in the initialization
        if b == max_trn_batch:
            break
        b += 1

        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 200 == 0:
            print(f'epoch: {i: 2} batch: {b: 4} [{10 * b: 6}/8000] loss: {loss.item(): 10.8f} \ acc: {trn_corr.item() * 100/ (10 * b):7.3f}%')
        
    train_losses.append(loss.item())
    train_correct.append(trn_corr)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # limit the number of batches
            if b == max_tst_batch:
                break

            y_val = CNNmodel(X_test)

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    
    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time: .0f} seconds')

# saving the model and analysis
torch.save(CNNmodel.state_dict(), f'./models/model.S.{VERSION}.{int(time.time())}.pt')
pd.DataFrame(np.array(train_losses)).to_csv('./data/interim/train_losses.csv', header = None, index = False)
pd.DataFrame(np.array(test_losses)).to_csv('./data/interim/test_losses.csv', header = None, index = False)
pd.DataFrame(np.array(train_correct)).to_csv('./data/interim/train_correct.csv', header = None, index = False)
pd.DataFrame(np.array(test_correct)).to_csv('./data/interim/test_correct.csv', header = None, index = False)
