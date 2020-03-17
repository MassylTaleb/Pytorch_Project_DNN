#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# In[3]:


data = pd.read_csv("Dev/PythonDeepLearning/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/diabetes.csv")


# In[4]:


data


# In[5]:


x = data.iloc[:, 0:-1].values
y_string = list(data.iloc[:, -1])


# In[6]:


x.shape


# In[8]:


len(y_string)


# In[9]:


y_int = []
for s in y_string:
    if s == 'positive':
        y_int.append(1)
    else:
        y_int.append(0)


# In[10]:


y_int


# In[11]:


y = np.array(y_int, dtype='float64')


# In[13]:


x


# In[14]:


sc = StandardScaler()
x = sc.fit_transform(x)


# In[15]:


x


# In[16]:


x = torch.tensor(x)
y = torch.tensor(y)


# In[18]:


y = y.unsqueeze(1)


# In[21]:


class Dataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)


# In[22]:


dataset = Dataset(x,y)
len(dataset)


# In[24]:


train_loader = torch.utils.data.DataLoader(dataset= dataset, batch_size = 32, shuffle = True)


# In[25]:


train_loader


# In[34]:


# Now let's build the above network
class Model(nn.Module):
    def __init__(self, input_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_features, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 3)
        self.fc4 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out


# In[35]:


net = Model(7,1)
criterion = torch.nn.BCELoss(size_average= True)
optimizer = torch.optim.SGD(net.parameters(), lr= 0.1, momentum = 0.9)


# In[37]:


num_epochs = 200
for epoch in range(num_epochs):
    for inputs,labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()
        # Feed Forward
        output = net(inputs)
        # Loss Calculation
        loss = criterion(output, labels)
        # Clear the gradient buffer (we don't want to accumulate gradients)
        optimizer.zero_grad()
        # Backpropagation 
        loss.backward()
        # Weight Update: w <-- w - lr * gradient
        optimizer.step()
        
    #Accuracy
    # Since we are using a sigmoid, we will need to perform some thresholding
    output = (output>0.5).float()
    # Accuracy: (output == labels).float().sum() / output.shape[0]
    accuracy = (output == labels).float().mean()
    # Print statistics 
    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss, accuracy))


# In[ ]:




