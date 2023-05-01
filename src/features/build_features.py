# importing libraries
import os
import warnings

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid, save_image

warnings.filterwarnings("ignore")

# constants

# preprocessing
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

root = './data/raw/CATS_DOGS'
train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform = train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform = test_transform)

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = True)

class_names = train_data.classes

print(class_names)
print(f'Training images available: {len(train_data)}')
print(f'Testing images available: {len(test_data)}')

# displaying first 10

# for images, labels in train_loader:
#     print("checking contents")
#     print(images)
#     break

# print('Label:', labels.numpy())
# print('Class:', *np.array([class_names[i] for i in labels]))

# im = make_grid(images, nrow = 5)

# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.225]
# )
# im_inv = inv_normalize(im)

# plt.figure(figsize = (12,4))
# # plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
# # plt.show()

# saving the train_data and test data
def save_images(data_loader, path):
    for i, (images, labels) in enumerate(data_loader):
        for j, (image, label) in enumerate(zip(images, labels)):
            index = (i * 10) + j
            
            if(index % 2400 == 0):
                print(f"Printing progress: {index}")

            if(label == 0):
                save_image(image, f'{path}/CAT/image{index}.png')
            else:
                save_image(image, f'{path}/DOG/image{index}.png')                

save_images(train_loader, './data/processed/train')
# save_images(test_loader, './data/processed/test')
