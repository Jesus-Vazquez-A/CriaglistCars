#!/usr/bin/env python
# coding: utf-8

# ### *Import Libraries*

# In[1]:


import numpy as np
import joblib
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


def scalerInputs(x):
    scaler = joblib.load("C:\\Users\\amado\\Desktop\\app\\scaler.pkl")
    return scaler.transform(x)

def robustTarget(pred):
    robust = joblib.load("C:\\Users\\amado\\Desktop\\app\\robust.pkl")
    return robust.inverse_transform(pred)


# In[4]:


def loadJsonFile():
       
       with open("C:\\Users\\amado\\Desktop\\app\\columns.json") as F:
           
           json_file = json.loads(F.read())
           json_file = np.asarray(json_file['columns'])
           
           return json_file


# In[6]:


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.fc1 = nn.Linear(in_features = 56,out_features = 64)
        self.fc2 = nn.Linear(in_features = 64,out_features = 32)
        self.fc3 = nn.Linear(in_features= 32,out_features=16)
        self.output = nn.Linear(in_features = 16,out_features = 1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)
    
def loadModel():
    model = Model()
    model.to('cpu')
    model.load_state_dict(torch.load("C:\\Users\\amado\\Desktop\\app\\craiglist_model.pth",map_location ='cpu'))
    return model
    