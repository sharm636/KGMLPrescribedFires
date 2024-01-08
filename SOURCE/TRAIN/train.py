# IMPORT LIBRARIES

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras import backend as bk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, ConvLSTM3D, Conv3D, Masking, SpatialDropout3D, Bidirectional
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers.core import SpatialDropout2D
from keras import backend as K

import bp3d
import os
import sys
import s3fs
import zarr
import numpy as np
import pandas as pd


data_dir = '../DATA'
save_dir = '../DATA/RESULTS'
file = os.path.join(data_dir, 'indices.txt')

f=open(file,"r")
lines=f.readlines()
samples=[]
for x in lines:
    samples.append(x.split(' ')[0])
f.close()



# LOAD IN DATA

datasets = []
ys = []
test_datasets, test_ys = [], []
for i in range(100):
    if ((i <50)):
        datasets.append(
            np.load(os.path.join(
            data_dir, 'train-features-'+str(samples[i])+'.npy'
            ))
        )
        ys.append(
                    np.load(os.path.join(
            data_dir, 'train-y-'+str(samples[i])+'.npy'
            ))

        )
    elif ((i>=50)):
        test_datasets.append(
                    np.load(os.path.join(
            data_dir, 'test-features-'+str(samples[i])+'.npy'
            ))

        )
        test_ys.append(
                            np.load(os.path.join(
            data_dir, 'test-y-'+str(samples[i])+'.npy'
            ))

        )




x_train = datasets[0]
def cl_model():
    '''
     We construct 4 `ConvLSTM2D` layers,
     followed by a `Conv3D` layer for 
     the spatiotemporal model.
    '''
    inp = layers.Input(shape=(None, *x_train.shape[2:]))
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(10, 10),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu", padding="same"
    )(x)
    model = keras.models.Model(inp, x)
    model0 = model
    return model0




# LOSS Next Steps
# 1. Fuel Transport: does not increase (yt- yt-1 <=0)
# penalize positive difference between yt and y t-1
# 2. Spread consistency: fire metrics help regularize 
# ROS, BI, BA estimates that are far from ground truth
# 3. Burn Regularization : penalize over-burning or under-burning based on the 
# burning / susceptibility in ground truth fuel density


# We define helper functions that are used to estimate the spread consistency metrics
def ROS_BA_Calculations(arr):
    '''
    Essentially the main function here. Contains the calls for rate of spread
    and burned area calculations. Hardcoded dx/dy/timestep can be changed here.
    Note that the timestep is the timestep of output frequency.

    Inputs:
        arr: np.array[ntimes,nx,ny,nz] , Fuel density array 
        
    Outputs: 
        None
    '''
    ntimes,_,nx,ny,nz = arr.shape
    dx,dy,timestep = 2.,2.,50.
    
#     a=calculate_instantaneous_ROS(arr,[dx,dy],timestep)
    b=calculate_average_ROS(arr,[dx,dy],timestep)
    c=calculate_BA(arr,[dx,dy])
#     d=calculate_BA(arr,[dx,dy],units='m2')        
#     e=calculate_BA(arr,[dx,dy],units='acres')
    return [b,c]
  
    
def calculate_instantaneous_ROS(arr,cell_size,timestep,ign_loc_ymin=None):
    '''
    Calculates the instantaenous rate of spread for each time step, defined as 
    the distance travelled over the time spent to get there from the last timestep
    to the current one.
    
    
    Inputs:
        mdl: np.array[ntimes] , List of maximum downwind index locations for each timestep
        cell_size [float,float] , Cell size in meters for both the x and y directions
        timestep (float) , Timestep of output frequency for fire related files in sec.
        ign_loc_ymin (float), index of the minimum y index of the original location. 
    
    Outputs:
        ros_inst [ntimes] , List of instantaneous rate of spread values in m/s
    '''
    dx,dy = cell_size
    mdl = calculate_max_downwind_loc(arr,ign_loc_ymin=None)
    ros_inst = tf.experimental.numpy.zeros_like(mdl)
    ros_inst_0 = tf.experimental.numpy.zeros_like(mdl[0])
    ros_inst = tf.experimental.numpy.diff(mdl)*dy / timestep

    ros_inst = tf.concat([ros_inst_0[tf.newaxis,tf.newaxis], ros_inst[:,tf.newaxis]],axis=0)    
    return ros_inst

def calculate_average_ROS(arr,cell_size,timestep,ign_loc_ymin=None):
    '''
    Calculates the average rate of spread for the fire for each time step, defined as 
    the distance travelled over the time spent to get there from the first timestep
    to the current one.
    
    
    Inputs:
        mdl: np.array[ntimes] , List of maximum downwind index locations for each timestep
        cell_size [float,float] , Cell size in meters for both the x and y directions
        timestep (float) , Timestep of output frequency for fire related files in sec.
        ign_loc_ymin (float), index of the minimum y index of the original location. 
    
    Outputs:
        ros_ave [ntimes] , List of average rate of spread values in m/s
    '''
    dx,dy = cell_size
    mdl = tf.cast(calculate_max_downwind_loc(arr,ign_loc_ymin=None), tf.float32)
    ros_ave = tf.experimental.numpy.zeros_like(mdl)
    ros_ave_0 = tf.experimental.numpy.zeros_like(mdl[0])
    ros_ave = tf.map_fn(
        lambda x: (x - ros_ave_0)*dy / (timestep), 
        mdl[1:,...]
    )
    ros_ave = tf.divide(ros_ave, tf.range(1.,ros_ave.shape[0]+1.))

    ros_ave = tf.concat([ros_ave_0[tf.newaxis,tf.newaxis], ros_ave[:,tf.newaxis]],axis=0)    
    return ros_ave

def calculate_BA(arr,cell_size,units='perc'):
    '''
    Calculates the total burned area for the fire at each time step. This can be outputted
    as either a percentage of the total burn unit, in m^2, or in acres. This is analyzed just
    in the surface fuels
    
    Inputs:
        mdl: np.array[ntimes] , List of maximum downwind index locations for each timestep
        cell_size [float,float] , Cell size in meters for both the x and y directions
        units (string) , 
    
    Outputs:
        burned_area [ntimes] , List of total burned area metric for each timestep
    '''
    dx,dy = cell_size
    ntimes,_,nx,ny,nz = arr.shape
    m2ac = 0.000247105
    bi = fueldens_to_burnindex(arr)
    burned_area = tf.zeros(ntimes)
    # This 796 subtraction is to account for the edges of the domain. 
    burned_area = tf.map_fn(
        lambda x: len(
            tf.experimental.numpy.where(x[:,:,:,0]==1)[1]
        ) + len(
            tf.experimental.numpy.where(x[:,:,:,0]==-1)[1]
        ) - 796,
        bi,
        parallel_iterations=True
    )
    if units == 'perc':
        burned_area = burned_area / (nx*ny)
    elif units == 'm2':
        burned_area = burned_area * dx *dy
    elif units == 'acres':
        burned_area = burned_area * dx *dy * m2ac
    return burned_area

def calculate_max_downwind_loc(arr,ign_loc_ymin=None):
    '''
    This function is a helper function that calculates the furthest downwind 
    spread of a particular array. It assumes you're giving the full fuel density
    4-dimensional array and assumes that the maximal spread is in the 
    eastern direction (i.e., assumes a mostly western wind)
    
    Since at time zero, there are no ignitions, we can set the minimum ign location
    on the y axis to the initial timestep, which will facilitate the calculation
    of average and instantaneous ROS.
    
    Finding first nonzero index is described here:
    https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array#:~:text=To%20find%20the%20first%20zeros,for%20use%20in%20the%20function.
        
    
    Inputs:
        arr: np.array[ntimes,nx,ny,nz] , Fuel density array 
        ign_loc_ymin [float], index of the minimum y index of the original location. 
        
    Outputs:
        mdl: np.array[ntimes] , List of maximum downwind index locations for each timestep
    '''
    
    ntimes, _,nx,ny,nz = arr.shape
    bi = tf.cast(fueldens_to_burnindex(arr), tf.int64)
    mdl = tf.zeros(ntimes)
    if ign_loc_ymin==None:
        sums = tf.experimental.numpy.sum(bi[0,0,:,:,0], axis=0)+2 # Looking at just the surface fuels 
#         sums[0],sums[-1] = 0.,0. # This is done since the first and last rows/columns are immediately zeroes. 
        ign_loc_ymin= tf.math.argmax(sums!=0,axis=0)      
    mdl = tf.map_fn(
        lambda x: (
            ny-1
        ) - tf.math.argmax(
            tf.reduce_sum(x[0,:,:,0]+2, axis=0)[::-1]!=0, 
            axis=0
        ), 
        elems= bi,
        parallel_iterations=True
    )

        
    return mdl

def fueldens_to_burnindex(arr):
    '''
    Helper function that calculates for all time steps and layers whether the cells are:
        burnt (-1), burning (1), unperturbed (0)
        
    Need to add  a functionality for places that had no fuel from the beginning?

    Inputs:
        arr: np.array[ntimes,nx,ny,nz] , Fuel density array 
        
    Outputs:
        arr: np.array[ntimes,nx,ny,nz] , Burnt/Unburnt index for the entire array
    '''
    
    temp =tf.experimental.numpy.divide(arr, arr[0,...][tf.newaxis,...]+1e-5) 
    burn_index = tf.experimental.numpy.ones_like(temp)
    burn_index *= 1
    burn_index = tf.where(temp> 0.99, 0, tf.where(temp<0.05, -1, tf.where(((temp>0.05)&(temp<0.99)), 1, -2)))

    return burn_index


# The Physic-guided loss 
class WildFirePhysicsLoss(tf.keras.losses.Loss):
  def __init__(self, reduction='none', constraint = 0, unburned = 0, burned = 0, firemetrics=0):
    self.name = 'physics guided loss'
    self.loss = tf.nn.l2_loss
    self.reduction = reduction
    self.firemetrics = firemetrics
    self.constraint = constraint
    self.unburned = unburned
    self.burned = burned

  def __call__(self, y_true, y_pred, sample_weight=None):

    # MSE LOSS.
    loss = (y_true - y_pred)**2 


    mask = tf.cast((y_pred[1:,:,:,:,0] - y_pred[:-1,:,:,:,0]) > 0, tf.float32)
    PHYSICS_LOSS = loss[1:]
    PHYSICS_LOSS = tf.boolean_mask(PHYSICS_LOSS, mask)

#     # BURNED.
    mask = tf.cast(y_true[:,:,:,:,0] < 1e-2, tf.float32)
    BURNED_LOSS = loss
    BURNED_LOSS = tf.boolean_mask(BURNED_LOSS, mask)

    # UNBURNED.
    mask = tf.cast(y_true[:,:,:,:,0] > 1e-2, tf.float32)
    UNBURNED_LOSS = loss
    UNBURNED_LOSS = tf.boolean_mask(UNBURNED_LOSS, mask)    
    # ROS, BI, BA. 
    if self.firemetrics>0:
        true_list = ROS_BA_Calculations(y_true)
        pred_list = ROS_BA_Calculations(y_pred)
        FIREMETRICS_LOSS = ((pred_list[0] - true_list[0])**2) + ((tf.cast(pred_list[1], tf.float32) - tf.cast(true_list[1], tf.float32))**2)
        loss_final = tf.reduce_mean(
            loss
        ) + self.constraint *  tf.cond(
              tf.equal(tf.size(PHYSICS_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(PHYSICS_LOSS)
            
          ) + self.burned * tf.cond(
              tf.equal(tf.size(BURNED_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(BURNED_LOSS)
            
          ) + self.unburned * tf.cond(
              tf.equal(tf.size(UNBURNED_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(UNBURNED_LOSS)
          
          ) + self.firemetrics *tf.cond(
              tf.equal(tf.size(FIREMETRICS_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(FIREMETRICS_LOSS)
           
          )
    else:
        loss_final = tf.reduce_mean(
            loss
        ) + self.constraint *  tf.cond(
              tf.equal(tf.size(PHYSICS_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(PHYSICS_LOSS)
           
          ) + self.burned * tf.cond(
              tf.equal(tf.size(BURNED_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(BURNED_LOSS)
            
          ) + self.unburned * tf.cond(
              tf.equal(tf.size(UNBURNED_LOSS), 0), 
              lambda : tf.constant(0.0), lambda: tf.reduce_mean(UNBURNED_LOSS)
            
          )

    # Return the total.
    return loss_final

def plot_static(y_true, y_pred, filename):
    test_points = [0, 10, 20, 30, 49]
    test_point_index =  test_points

    fig, axs = plt.subplots(2, 5,figsize=(50, 10))

    counter1, counter2 = 0,0
    for i, ax in enumerate(axs.flatten()):
        if i in range(5):
            ax.set_title(f"Predicted: t={test_points[counter1]}")
            ax.imshow(y_pred[test_point_index[counter1],...].squeeze().astype('float64'),cmap='rainbow',origin="lower")
            counter1+=1
        if i in range(5, 11):
            ax.set_title(f"Observed: t={test_points[counter2]}")
            ax.imshow(y_true[test_point_index[counter2],...].squeeze().astype('float64'),cmap='rainbow',origin="lower")
            counter2+=1

    fig.tight_layout() 
    plt.savefig(filename+'.png',dpi=100)

    
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


scenarios = [ [1e-3, 1e-3, 1e-3, 1e-3], [0, 0, 0, 1e-3], [0,0,0, 0], [1e-3, 0, 0, 0], [0, 1e-3, 0, 0], [0, 0, 1e-3, 0]]
names = ['all', 'fm', 'mse', 'pc', 'b', 'ub'] # all loss, fire metrics, mse only, physical constraint, burned, unburned

for count in range(len(scenarios))

    model = cl_model()
    model.compile(
        loss=WildFirePhysicsLoss(
            constraint=scenarios[count][0],
            burned = scenarios[count][1],
            unburned=scenarios[count][2], 
            firemetrics=scenarios[count][3]
        ),
        optimizer=keras.optimizers.Adam(lr=1e-3, clipvalue=1.0),
        metrics = [tf.keras.metrics.mean_squared_error]
    )

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10)

    epochs = 5
    batch_size = 10

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=100*batch_size)

    model.save_weights(checkpoint_path.format(epoch=0))

    # Fit the model to the training data.
    for i in range(len(datasets)): 
        model.fit(
            datasets[i],
            ys[i],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_datasets[0], test_ys[0]),
            callbacks=[early_stopping, reduce_lr, cp_callback],
            verbose=2
        )
    model.save_weights("cp-final-scenario-"+str(names[count])+".ckpt")
