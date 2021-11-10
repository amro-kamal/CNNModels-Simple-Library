# cnnModels
**A simple Pytorch library to implement and train some of the famous CNNs architectures.**

In this simple lib I implemented some of the famous CNN architectures in pytorch. You can use the lib to to train any of these models on your custom data using few lines of code. The library supports cpu training and single-gpu training.

Now the library supports 4 models:
-[x] **VGG16,** 
-[x] **ResNet,**
-[x] **InceptionV1
-[x] **Alexnet**

## How to use the library: 
First prepare your data using pytorch dataloader. Here I will use cifar10 dataset from torchvision.
 ```python
import troch.nn as nn
from torchvision.datasets import CIFAR10 
from torchvision import transforms
from torch.utils.data import DataLoader
mytransforms = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize(224),
                   transforms.Normalize((0,0,0),(255.0 ,255.0 , 255.0)) 
                               ])
cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=mytransforms)
cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=mytransforms

train_loader = DataLoader(cifar10_train , batch_size=64 , shuffle=True)
test_loader = DataLoader(cifar10_test , batch_size=64)

num_classes=10

```
Then create the model cnnModels library, the optimizer and define the loss function

```python

from cnnModels.Alexnet import AlexNet
alexnet = AlexNet(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(alexnet.parameters(), lr=1e-4)

```
Lets check for the gpu 
```python
 device = torch.device('cuda') if torch.cuda.is_available() else  torch.device('cpu')
```
Now train AlexNet you need just one line of code

```python
alexnet.Train(device ,criterion, train_loader, test_loader , optimizer, num_epochs=10)
```
