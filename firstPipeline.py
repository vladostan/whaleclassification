#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import datetime
import sys

# In[ ]:

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

# Print stdout to file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

print('Date and time: {}\n'.format(loggername))

# In[ ]:
datadir = "data/"
os.listdir(datadir)


# In[ ]:

train_df = pd.read_csv(datadir + "train.csv")
train_df.head()


# In[ ]:
rows, cols = train_df.shape
print("Entries: {}, columns: {}".format(rows,cols))


# In[ ]:
num_classes = np.count_nonzero(np.unique(train_df.values[:,1]))
print(num_classes)


# In[ ]:


sortd = train_df.groupby("Id").size().sort_values()
sortd.tail()


# In[ ]:
#Class weighting
cw = np.median(sortd.values)/sortd.values
print(cw)

# In[ ]:


from keras.applications.xception import Xception

premodel = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3), classes=num_classes)
premodel.summary()


# In[ ]:


# for layer in premodel.layers[:5]:
#     layer.trainable = False


# In[ ]:


from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dense

#Adding custom Layers 
# x = model.output
# x = Flatten()(x)
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)
# x = Dense(1024, activation="relu")(x)
# predictions = Dense(16, activation="softmax")(x)

x = premodel.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation="softmax")(x)


# In[ ]:


from keras.models import Model

model = Model(input = premodel.input, output = predictions)

del(premodel)

model.summary()


# In[ ]:


from keras import optimizers
from metrics import dice_loss, dice

learning_rate = 1e-3
optimizer = optimizers.Adam(lr = learning_rate)
metrics = ['accuracy']
loss = 'categorical_crossentropy'
# metrics = [dice]
# loss = [dice_loss]

model.compile(loss = loss, optimizer = optimizer, metrics=metrics)


# In[ ]:


from keras import callbacks


model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience = 3, verbose = 1, min_lr=1e-7)
csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
early_stopper = callbacks.EarlyStopping(monitor='loss', min_delta = 0.01, patience = 5, verbose = 1)

callbacks = [model_checkpoint, reduce_lr, csv_logger, early_stopper]


# In[ ]:


from keras.applications.xception import preprocess_input
from keras.preprocessing import image

def get_np_image(df, target_size):
    
    img = image.load_img("data/"+"train/"+df, target_size=target_size)
    x = image.img_to_array(img)
    x = preprocess_input(x)
    
    return(x)


# In[ ]:


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


# In[ ]:


y_encoded, label_encoder = encode(train_df["Id"])
print(y_encoded.shape)


# In[ ]:


x = get_np_image(df = train_df["Image"][0], target_size=(299,299,3))
print(x.shape)
print(x.dtype)


# In[ ]:


y = y_encoded[0]
print(y.shape)
print(y.dtype)


# In[ ]:


# def generate_data(directory, batch_size):
#     """Replaces Keras' native ImageDataGenerator."""
#     i = 0
#     file_list = os.listdir(directory)
#     while True:
#         image_batch = []
#         for b in range(batch_size):
#             if i == len(file_list):
#                 i = 0
#                 random.shuffle(file_list)
#             sample = file_list[i]
#             i += 1
#             image = cv2.resize(cv2.imread(sample[0]), INPUT_SHAPE)
#             image_batch.append((image.astype(float) - 128) / 128)

#         yield np.array(image_batch)

def custom_generator(df, y_encoded, target_size = (299,299,3), batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        x_batch = []
        y_batch = []
                
        for b in range(batch_size):
            if i == len(df):
                i = 0
                
            x = get_np_image(df = train_df["Image"][i], target_size = target_size)
            y = y_encoded[i]
            
            i += 1
            
            x_batch.append(x)
            y_batch.append(y)
            
        yield (x_batch,y_batch)


# In[ ]:


batch_size = 4
steps_per_epoch = len(train_df)//batch_size
# validation_steps = len(train_df)//batch_size
epochs = 100
verbose = 2

print("Batch size: {}, epochs: {}".format(batch_size, epochs))

# In[ ]:


train_gen = custom_generator(df=train_df, y_encoded=y_encoded, batch_size=batch_size)


# In[ ]:


history = model.fit_generator(
    train_gen,
    steps_per_epoch = steps_per_epoch,
    class_weight = cw,
    epochs = epochs,
    verbose = verbose,
    callbacks = callbacks
)

