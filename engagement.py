#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses
from tensorflow.keras.applications.resnet50 import ResNet50


# In[2]:


from daisee_data_preprocessing import DataPreprocessing
import datetime
import os
from tqdm import tqdm


# In[3]:


BATCH_SIZE = 32
LR = 0.005
EPOCHS = 100


# In[4]:


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = 'checkpoints/'
log_dir = 'logs/'


# In[5]:


### This part works for setting up space in gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 1 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# In[6]:


preprocessing_class = DataPreprocessing()


# In[7]:


# Open train set
tfrecord_path = 'tfrecords/train.tfrecords'
train_set = tf.data.TFRecordDataset(tfrecord_path)
# Parse the record into tensors with map.
train_set = train_set.map(preprocessing_class.decode)
train_set = train_set.shuffle(1)
train_set = train_set.batch(BATCH_SIZE)


# In[8]:


# Open test set
tfrecord_path = 'tfrecords/test.tfrecords'
test_set = tf.data.TFRecordDataset(tfrecord_path)
# Parse the record into tensors with map.
test_set = test_set.map(preprocessing_class.decode)
test_set = test_set.shuffle(1)
test_set = test_set.batch(BATCH_SIZE)


# In[9]:


# Open val set
tfrecord_path = 'tfrecords/val.tfrecords'
val_set = tf.data.TFRecordDataset(tfrecord_path)
# Parse the record into tensors with map.
val_set = val_set.map(preprocessing_class.decode)
val_set = val_set.shuffle(1)
val_set = val_set.batch(BATCH_SIZE)


# In[10]:


def create_log_dir(log_dir, checkpoint_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)


# In[11]:


def network():
    model = tf.keras.Sequential()
    model.add(kl.InputLayer(input_shape=(224, 224, 3)))
    # First conv block
    model.add(kl.Conv2D(filters = 96, kernel_size=7, padding='same', strides=2))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(3, 3)))
    # Second conv block
    model.add(kl.Conv2D(filters = 256, kernel_size=5, padding='same', strides=1))
    model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    # Third-Fourth-Fifth conv block
    for i in range(3):
        model.add(kl.Conv2D(filters = 512, kernel_size=3, padding='same', strides=1))
        model.add(tf.keras.layers.ReLU())
    model.add(kl.MaxPooling2D(pool_size=(3, 3)))
    # Flatten
    model.add(kl.Flatten())
    # First FC 
    model.add(kl.Dense(4048))
    # Second Fc
    model.add(kl.Dense(4048))
    # Third FC
    model.add(kl.Dense(4))
    # Softmax at the end
    model.add(kl.Softmax())
    
    return model


# In[12]:


model = network()


# In[13]:


'''
https://keras.io/guides/writing_a_training_loop_from_scratch/
Compile into a static graph any function that take tensors as input to apply global performance optimizations.
'''


# In[14]:


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # Track progress
    train_loss_avg.update_state(loss_value)
    train_accuracy.update_state(y, logits)
    return loss_value


# In[15]:


@tf.function
def test_step(x, y, set_name):
    logits = model(x)
    if set_name == 'val':
        val_accuracy.update_state(y, logits)
    else:
        test_accuracy.update_state(y, logits)


# In[16]:


optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
create_log_dir(log_dir, checkpoint_dir)
train_summary_writer = tf.summary.create_file_writer(log_dir)


# In[17]:


train_loss_avg = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.MeanAbsoluteError()
val_accuracy = tf.keras.metrics.MeanAbsoluteError()
test_accuracy = tf.keras.metrics.MeanAbsoluteError()


# In[ ]:


for epoch in range(EPOCHS):
    # Training loop
    for x_batch_train, y_batch_train in tqdm(train_set, total=1517):
        # Do step
        loss_value = train_step(x_batch_train, y_batch_train)
        
    # Test on validation set
    for x_batch_val, y_batch_val in val_set:
        test_step(x_batch_val, y_batch_val, 'val')
    
    # Reset training metrics at the end of each epoch
    train_acc = train_accuracy.result()
    train_accuracy.reset_states()
    val_acc = val_accuracy.result()
    val_accuracy.reset_states()
    
    with train_summary_writer.as_default():
        tf.summary.scalar('Train loss', train_loss_avg.result(), step=epoch)
        tf.summary.scalar('Train MAE', train_acc, step=epoch)
        tf.summary.scalar('Val MAE', val_acc, step=epoch)
        
    if epoch % 10 == 0:
        tf.keras.models.save_model(model, '{}/Epoch_{}_model.hp5'.format(checkpoint_dir, str(epoch)), save_format="h5")


# In[ ]:


# Test on validation set
for x_batch_test, y_batch_test in test_set:
    test_step(x_batch_test, y_batch_test, 'test')
test_set_acc = test_accuracy.result()

