#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import torch
import warnings
import streamlit as st
import utils


# In[11]:


warnings.filterwarnings('ignore')


# In[12]:


columns = utils.loadJsonColumns()
model = utils.loadModel()
model_names = utils.loadJsonModels()


# In[13]:

st.cache(allow_output_mutation=True)

def UserInputs():
 
    manufacturer_list = ['audi', 'bmw', 'cadillac',
       'chevrolet', 'chrysler', 'dodge', 'ford', 'gmc', 'honda',
       'hyundai', 'infiniti', 'jeep', 'kia', 'lexus', 'mazda',
       'mercedes-benz', 'mitsubishi', 'nissan', 'subaru', 'acura',
       'toyota', 'volkswagen']

    
    manufacturer =  st.selectbox("Manufacturer: ", manufacturer_list)
    model_car =  st.selectbox("Model: ", model_names)
    body = st.selectbox("Type Body: ", ['sedan','crossover','hatchback','SUV','coupe','pickup','mini-van','fastback','convertible'])
    year = st.number_input('Year: ',min_value = 2000,max_value = 2024,step = 1)
    odometer = st.number_input("Mileage: ", min_value=0,max_value = 200000)
    cyl = st.number_input("Cylinders: ", min_value = 0,max_value=8, step=1)
    engine = st.number_input("Engine: ",min_value=0.0,max_value = 6.2,step = 0.1)
    fuel = st.selectbox("Fuel: ", ["gas", "diesel", "electric", "hybrid"])
    transmission = st.selectbox("Transmission: ", ["automatic", "manual"])
    hp = st.number_input("HP: ", min_value=1, step=1)
    title_status = st.selectbox("Title Status: ", ["missing","lein","salvage","rebuilt","clean"])
    

    return manufacturer,model_car,body,year,odometer,cyl,engine,fuel,transmission,hp,title_status


# In[14]:


def preprocessing():

    manufacturer,model_car,body,year,odometer,cyl,engine,fuel,transmission,hp,title_status = UserInputs()
    
    model_car = model_car.lower()
    
    zeros = np.zeros(len(columns))
    title_status_dict = {'missing':0,'lien':1,'salvage':2,'rebuilt':3,'clean':4}
    zeros[0] = year
    zeros[1] = cyl
    zeros[2] = np.sqrt(odometer)
    zeros[3] = title_status_dict[title_status]
    zeros[4] = np.where(transmission=="automatic",1,0)
    zeros[5] = engine
    zeros[6] = hp
    premium_list =  ['gmc','cadillac','audi','bmw','mercedes-benz','infiniti','acura','lexus']
    if manufacturer in premium_list:
        zeros[7] = 1
    else:
        zeros[7] = 0


    manufacturer_idx = np.where(manufacturer == columns)[0][0]
    fuel_idx = np.where(fuel == columns)[0][0]
    type_body_idx = np.where(body == columns)[0][0]
    country_idx = np.where(country == columns)[0][0]
    model_idx = np.where(model_car == columns)[0][0]

    if manufacturer_idx >= 0:
        zeros[manufacturer_idx] = 1


    if fuel_idx >=0:
        zeros[fuel_idx] = 1


    if type_body_idx>=0:
        zeros[type_body_idx] = 1

    if country_idx>=0:
        zeros[country_idx] = 1


    if model_idx>=0:
        zeros[model_idx] = 1


    american = ['ford','chevrolet','jeep','dodge','chrysler','cadillac','gmc','tesla']
    japan = ['nissan','honda','toyota','mazda','mitsubishi','subaru','infiniti','acura','lexus']
    germany = ['bmw','mercedes-benz','audi','volkswagen']
    south_korea = ['kia','hyundai']
    
    if manufacturer in germany:
        
        zeros[421] = 1
    
    if manufacturer in japan:
        zeros[422] = 1
        
    if manufacturer in south_korea:
        zeros[423] = 1
    
    if manufacturer in american:
        zeros[424] = 1

 

    zeros = np.asarray([zeros])


    zeros = utils.scalerInputs(zeros)
    zeros = torch.from_numpy(zeros).float()




    return zeros


# In[18]:


def predict(newdata):
    
    pred =  model(newdata).float()
    pred = pred.detach().numpy()
    return utils.robustTarget(pred)


# In[20]:


st.cache(allow_output_mutation=True)

def main():
    st.write(""" # Vehicle evaluation """)
    newdata = preprocessing()

    
    if st.button(label = 'Predict'):

        
        price=predict(newdata)
        st.success(f'The estimated price of the vehicle is: $ {price} USD')



if __name__ == '__main__':
    main()
