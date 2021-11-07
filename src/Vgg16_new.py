import numpy as np
import torch
import torchvision
import time
#import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
import itertools  
import torch.nn.functional as F
from PIL import Image
import splitfolders
from sklearn.metrics import f1_score
from   core_qnn.quaternion_ops import *
from   core_qnn.quaternion_layers import *
#splitfolders.ratio('asphalt', output="asphalt", seed=1337, ratio=(.8, 0.1,0.1)) 
#device = torch.cuda.set_device(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
transform_test = transforms.Compose([transforms.Resize(227), 
                                 transforms.ToTensor()])
transform = transforms.Compose([ transforms.ToTensor()])
dataset = datasets.ImageFolder('building_train_0.7k', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
#testdata = datasets.ImageFolder('building_test_28k', transform=transform)
testdata = datasets.ImageFolder('asphalt_test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=10, shuffle=True)
valdata = datasets.ImageFolder('building_val_0.7k', transform=transform)
val_loader = torch.utils.data.DataLoader(valdata, batch_size=10, shuffle=True)

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])

        self.fcs = nn.Sequential(
           nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)
cnn = VGG_net(in_channels=3, num_classes=1).to(device)
pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(pytorch_total_params)
cnn.to(device)
optimizer = optim.SGD(cnn.parameters(), lr=0.01,momentum=0)  
criterion = torch.nn.BCELoss()
loss_array=[]
acc_array=[]
val_loss_array = []
epochs=10
for epoch in range(epochs):
    start_time = time.time()
    running_loss=0
    running_accuracy=0
    validation_accuracy=0
    val_loss = 0
    cnn.train()
    for i,(images, labels) in tqdm(enumerate(train_loader)):  ### label 1 = crack || label 0 = no crack
        #print(labels[0])
        #plt.imshow(images[0].permute(1, 2, 0),cmap='gray')
        #plt.show()
        images, labels = images.to(device), labels.to(device)
        images, labels =images.to(torch.float32) , labels.to(torch.float32)
        images = Variable(images)
        labels =Variable(labels)
         # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass
        outputs=cnn(images)
        # calculate-the-batch-loss
        loss = criterion(outputs.squeeze(),labels)
        # backward-pass:
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        #print(outputs)
        #print(outputs.shape)
        best_output = torch.round(outputs)
        #print(best_output)
        #accuracy = (best_output == labels).float().mean()
        correct_out=0
        for i in range(len(best_output)):

            if(best_output[i] == labels[i]):
                correct_out+=1
        accuracy = correct_out /10
        #loss = loss.mean()
        
        
        running_loss += loss.item()
        #running_accuracy+=accuracy.item()
        running_accuracy+=accuracy
        #optimizer.zero_grad()
        #loss_array.append(loss.item())
        #acc_array.append(accuracy.item())
        #if (i+1) % 100 ==0:
    cnn.eval()
    for i,(images, labels) in tqdm(enumerate(val_loader)):
        images, labels = images.to(device), labels.to(device)
        images, labels =images.to(torch.float32) , labels.to(torch.float32)
        images = Variable(images)
        labels =Variable(labels)
        outputs=cnn(images)
        #print(outputs)
        loss = criterion(outputs.squeeze(),labels)
        best_output = torch.round(outputs)
        #print(best_output)
        #print(labels)
        #accuracy = (best_output == labels).float().mean()
        
        #loss = loss.mean()
        correct_out=0
        for i in range(len(best_output)):

            if(best_output[i] == labels[i]):
                correct_out+=1
        accuracy = correct_out /10
        #validation_accuracy +=accuracy.item()
        validation_accuracy +=accuracy
        val_loss += loss.item()
        #val_loss_array.append(loss.item())
    print("Epoch {%d} - Training loss: {%f} - Val loss{%f} -Training Accuracy{%f} - Validation Accuracy{%f} "%(epoch+1,running_loss/len(train_loader),       
                                                                              val_loss/len(val_loader),running_accuracy/len(train_loader),validation_accuracy/len(val_loader)))
    print("Epoch execution time %s seconds " % (time.time() - start_time))                                                                          
    
cnn.eval()
correct = 0
total = 0
counter =0
total_f1 = 0
for images, labels in test_loader:
    #start_time = time.time()
    images, labels = images.to(device), labels.to(device)
    images = Variable(images)
    outputs = cnn(images)
    #print(outputs)
    predicted = torch.round(outputs.data)
    #print(predicted)
    #_, predicted = torch.max(outputs.data, 1)
    #print(predicted)
    #print(labels)
    total += labels.size(0)
    for i in range(len(predicted)):

        if(predicted[i] == labels[i]):
            correct+=1
    #correct += (predicted == labels).sum()
    total_f1 +=f1_score(labels.cpu(),predicted.cpu())
    counter+=1
    #print("Image execution time prediction %s seconds " % ((time.time() - start_time)/10))  ##execution time per batch
print(correct)   
print('test Accuracy %d test images: %f' % (total, correct/total))
print ('F1 score:', total_f1/counter)                                                                         



