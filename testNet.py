#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg


# In[2]:


test = os.listdir("data/test/")
print(len(test))


# In[3]:


col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''


# In[4]:


from keras.applications.xception import preprocess_input
from keras.preprocessing import image

def get_np_image(df, target_size = (299,299,3)):
    
    img = image.load_img("data/"+"test/"+df, target_size=target_size)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    
    return(x)


# In[5]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def encode(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded.astype(np.int32), label_encoder


# In[7]:


train_df = pd.read_csv("data/train.csv")
num_classes = np.count_nonzero(np.unique(train_df.values[:,1]))


# In[8]:


_ , label_encoder = encode(train_df["Id"])
label_encoder


# In[9]:


from keras.applications.xception import Xception
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Model

premodel = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), classes=num_classes)

x = premodel.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(input = premodel.input, output = predictions)

del(premodel)


# In[10]:


from keras import optimizers
from metrics import dice_loss, dice

learning_rate = 1e-3
optimizer = optimizers.Adam(lr = learning_rate)
metrics = ['accuracy']
loss = 'categorical_crossentropy'

model.compile(loss = loss, optimizer = optimizer, metrics=metrics)
model.load_weights("weights/2018-12-02 01-22-15.hdf5")


# In[11]:


def custom_generator(df, target_size = (299,299,3), batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        x_batch = []
                
        for b in range(batch_size):
            if i == len(df):
                i = 0
                
            x = get_np_image(df = df["Image"][i], target_size = target_size)
            
            i += 1
            
            x_batch.append(x)
            
        yield np.array(x_batch)


# In[12]:


batch_size = 1
steps_per_epoch = len(test_df)//batch_size
test_gen = custom_generator(df=test_df, batch_size=batch_size)


# In[ ]:


y_pred = model.predict_generator(test_gen,steps_per_epoch, verbose=1)


# In[ ]:


for i, pred in enumerate(y_pred):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_df.to_csv('submission.csv', index=False)

