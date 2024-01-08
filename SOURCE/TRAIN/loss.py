import tensorflow as tf
from tensorflow import keras
import numpy as np




class DMSE(tf.keras.losses.Loss):
  def __init__(self, reduction='none'):
    self.name = 'Dynamic MSE'
    self.loss = tf.keras.losses.MeanSquaredError
    self.reduction = reduction

  def __call__(self, y_true, y_pred, sample_weight=None):
    y_true_diff, y_pred_diff = tf.experimental.numpy.diff(y_true[:,:,:,:,0]), tf.experimental.numpy.diff(y_pred[:,:,:,:,0])
    loss = tf.nn.relu(y_pred_diff)

    # Mask off the losses on padding.
    mask = tf.cast(y_true_diff > 1e-5, tf.float32)
    loss *= mask

    # Return the total.
    return tf.reduce_mean(loss)


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
    
# The Physic-guided loss with the PGM structure
class WildFirePhysicsLoss(tf.keras.losses.Loss):
  def __init__(self, reduction='none', constraint = 0, unburned = 0, burned = 0, firemetrics=0):
    self.name = 'pgcl+ loss'
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
    prior_loss= 1e-5 * tf.reduce_mean((
        tfd.Poisson(tf.reduce_sum(
            tf.abs(tf.cast(fueldens_to_burnindex(y_true[:,0,:,:,0]), tf.float32)), axis=(1,2)
        )).sample()- tfd.Poisson(tf.reduce_sum(
            tf.abs(tf.cast(fueldens_to_burnindex(y_pred[:,0,:,:,0]), tf.float32)), axis=(1,2)
        )).sample()
    )**2)
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
           
          ) + prior_loss
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
            
          ) + prior_loss

    # Return the total.
    return loss_final