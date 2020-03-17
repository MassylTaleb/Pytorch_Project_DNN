#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# In[3]:


input_size = 784
hidden_size = 400
out_size = 10
epochs = 10
batch_size = 100
learning_rate = 0.001


# In[4]:


train_dataset = datasets.MNIST(root='./Dev/PythonDeepLearning/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/'
                               ,train=True, transform=transforms.ToTensor(),
                              download=True)
test_dataset = datasets.MNIST(root='./Dev/PythonDeepLearning/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/'
                               ,train=False, transform=transforms.ToTensor())


# In[5]:


train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# In[8]:


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn. Linear(hidden_size, out_size)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# In[9]:


net = Net(input_size, hidden_size, out_size)
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# In[10]:


for i, (images, labels) in enumerate(train_loader):
    print(images.size())
    images = images.view(-1,784)
    print(images.size())


# In[20]:


for epoch in range(epochs):
    correct_train = 0
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):   
        images = images.view(-1, 28*28)    
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
            
        outputs = net(images)       
        _, predicted = torch.max(outputs.data, 1)                                              
        correct_train += (predicted == labels).sum() 
        loss = criterion(outputs, labels)                 
        running_loss += loss.item()
        optimizer.zero_grad() 
        loss.backward()                                   
        optimizer.step()                                 
        
    print('Epoch [{}/{}], Training Loss: {:.3f}, Training Accuracy: {:.3f}%'.format
          (epoch+1, epochs, running_loss/len(train_loader), (100*correct_train.double()/len(train_dataset))))
print("DONE TRAINING!")


# In[21]:


with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
        images = images.view(-1,28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data,1)
        correct += (predicted == labels).sum().item()
        
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / len(test_dataset)))


# In[ ]:





# In[ ]:




