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
dataset = datasets.ImageFolder('building_train_14k', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
#testdata = datasets.ImageFolder('building_test_28k', transform=transform)
testdata = datasets.ImageFolder('asphalt_test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=10, shuffle=True)
valdata = datasets.ImageFolder('building_val_14k', transform=transform)
val_loader = torch.utils.data.DataLoader(valdata, batch_size=10, shuffle=True)

class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QuaternionConv(4,64,kernel_size=3,stride=1) #
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = QuaternionConv(64,64,kernel_size=3,stride=1)#block1
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = QuaternionConv(64,128,kernel_size=3,stride=1) #
        self.batch3 = nn.BatchNorm2d(128)
        self.conv4 = QuaternionConv(128,128,kernel_size=3,stride=1) #block2 
        self.batch4 = nn.BatchNorm2d(128)
        self.conv5 = QuaternionConv(128,256,kernel_size=3,stride=1) #
        self.batch5 = nn.BatchNorm2d(256)
        self.conv6 = QuaternionConv(256,256,kernel_size=3,stride=1) #
        self.batch6 = nn.BatchNorm2d(256)
        self.conv7 = QuaternionConv(256,256,kernel_size=3,stride=1) #block3
        self.batch7 = nn.BatchNorm2d(256)
        self.conv8 = QuaternionConv(256,512,kernel_size=3,stride=1) #
        self.batch8 = nn.BatchNorm2d(512)
        self.conv9 = QuaternionConv(512,512,kernel_size=3,stride=1) #
        self.batch9 = nn.BatchNorm2d(512)
        self.conv10 = QuaternionConv(512,512,kernel_size=3,stride=1) # block4
        self.batch10 = nn.BatchNorm2d(512)
        self.conv11 = QuaternionConv(512,512,kernel_size=3,stride=1) #
        self.batch11 = nn.BatchNorm2d(512)
        self.conv12 = QuaternionConv(512,512,kernel_size=3,stride=1) #
        self.batch12 = nn.BatchNorm2d(512)
        self.conv13 = QuaternionConv(512,512,kernel_size=3,stride=1) # block5
        self.batch13 = nn.BatchNorm2d(512)
        #self.fc1 = nn.Linear(4608,1024) #output*(3*3) from paper  
        self.fc1 = QuaternionLinear(512,4096) #output*(3*3) from paper      
        self.fc2=QuaternionLinear(4096,4096)
        self.fc3=QuaternionLinear(4096,4)
        self.dropout1 = nn.Dropout(0.65)
        self.dropout2 = nn.Dropout(0.65)
        self
        #self.sm = nn.LogSoftmax(dim=1) 
        self.sigm=torch.nn.Sigmoid()
    def forward(self,x):
            
            x = F.relu(self.batch1(self.conv1(x)))
            x = F.relu(F.max_pool2d(self.batch2(self.conv2(x)),2))
            x = F.relu(self.batch3(self.conv3(x)))
            x = F.relu(F.max_pool2d(self.batch4(self.conv4(x)),2))
            x = F.relu(self.batch5(self.conv5(x)))
            x = F.relu(self.batch6(self.conv6(x)))
            x = F.relu(F.max_pool2d(self.batch7(self.conv7(x)),2))
            x = F.relu(self.batch8(self.conv8(x)))
            x = F.relu(self.batch9(self.conv9(x)))
            x = F.relu(F.max_pool2d(self.batch10(self.conv10(x)),2))
            x = F.relu(self.batch11(self.conv11(x)))
            x = F.relu(self.batch12(self.conv12(x)))
            x = F.relu(F.max_pool2d(self.batch13(self.conv13(x)),2))
            #print(x.shape)
            #x = x.view(10,4608) 
            x = x.view(10,512)    
            #print(x.shape)
            x = F.relu(self.fc1(x))
            
            x=self.dropout1(x)
            x = F.relu(self.fc2(x))
            x=self.dropout2(x)
            x=self.fc3(x)
            w = torch.sum(x, dim=1) # size = [nrow, 1]
            #print(w)
            x = self.sigm(w)
            #x = self.sigm(x)
           # x = self.sm(x)
            return x
cnn = Vgg16()
pytorch_total_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(pytorch_total_params)
cnn.to(device)
optimizer = optim.SGD(cnn.parameters(), lr=0.001,momentum=0)  
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
        counter=0
        
        temp = torch.empty(size=(10,4,227,227))
        for image in images:
            image=image.numpy()
            image = np.resize(image,(4,227,227))
            image[0][3][:][:]=0
            image = torch.from_numpy(image)
            temp[counter] = image
            counter+=1
        #print(labels[0])
        #plt.imshow(images[0].permute(1, 2, 0),cmap='gray')
        #plt.show()
        temp, labels = temp.to(device), labels.to(device)
        temp, labels =temp.to(torch.float32) , labels.to(torch.float32)
        temp = Variable(temp)
        labels =Variable(labels)
         # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass
        outputs=cnn(temp)
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
        counter=0
        
        temp = torch.empty(size=(10,4,227,227))
        for image in images:
            image=image.numpy()
            image = np.resize(image,(4,227,227))
            image[0][3][:][:]=0
            image = torch.from_numpy(image)
            temp[counter] = image
            counter+=1
        temp, labels = temp.to(device), labels.to(device)
        temp, labels =temp.to(torch.float32) , labels.to(torch.float32)
        temp = Variable(temp)
        labels =Variable(labels)
        outputs=cnn(temp)
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
    counter_image=0
        
    temp = torch.empty(size=(10,4,227,227))
    for image in images:
        image=image.numpy()
        image = np.resize(image,(4,227,227))
        image[0][3][:][:]=0
        image = torch.from_numpy(image)
        temp[counter_image] = image
        counter_image+=1
    #start_time = time.time()
    temp, labels = temp.to(device), labels.to(device)
    temp = Variable(temp)
    outputs = cnn(temp)
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



