# importing libraries
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid

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

# prep the model, using pretrained alexnet
AlexNetmodel = models.alexnet(pretrained = True)
# print(AlexNetmodel)

for param in AlexNetmodel.parameters(): # freezing the params
    param.requires_grad = False

torch.manual_seed(42)
AlexNetmodel.classifier = nn.Sequential(
    nn.Linear(9216, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 2),
    nn.LogSoftmax(dim = 1)
)
# print(AlexNetmodel)

# prepare criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(AlexNetmodel.classifier.parameters(), lr = 0.001)

start_time = time.time()

epochs = 1
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

        y_pred = AlexNetmodel(X_train)
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

            y_val = AlexNetmodel(X_test)

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    
    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time: .0f} seconds')

print(test_correct)
print(f'Test Accuracy: {test_correct[-1].item() * 100 / 3000: .3f}%')

x = 2019
# Inverse normalize the images
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
class_names = ['CAT', 'DOG']

im = inv_normalize(test_data[x][0])
plt.imshow(np.transpose(im.numpy()))
plt.show()

AlexNetmodel.eval()
with torch.no_grad():
    new_pred = AlexNetmodel(test_data[x][0].view(1, 3, 224, 224)).argmax()

print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')
