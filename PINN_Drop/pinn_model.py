## This module contains the model for modeling a sessile drop.

## Import necessary packages.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PINN_Drop.utils as utils

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import load_model
import numpy as np
from tqdm import trange

class PINNDrop(Model):
    '''
    This is the PINN model. Contains the MLP for approximating z-profile and the laplacian pressure.

    Attributes:
        space_width (float): The width of the grid.
        num_grids (int): Number of valid coordinates in the spatial domain.
        target_vol (float): The total volume of the liquid.
        len_scale (float): The length scale in relation to the total volume of the drop.
        TCL_height (float): The z-profile threshold for determining edge points of the droplet.
        model (Sequential): A sequence of dense layers used to predict the z-profile of the drop.
        laplace_pressure (Variable): A trainable parameter representing the Laplacian pressure, normalized to surface tension.
        args (dict): Dictionary containing hyperpameters for the water droplet.
        contact_mask_memory (Variable): A non-trainable variable that stores edge coordinates of the droplet.
        z_old_memory (Variable): A non-trainable variable that stores previous z-profiles.
    
    Methods:
        call (self, inputs): Given (x, y) coordinates, returns their respective z-profile.

    Usage:
        This is the PINN model used to generate the height at each (x, y) coordinate.
    '''

    def __init__(self, args):
        '''
        Args:
            space_width (float): The width of the grid.
            num_grids (int): The number of grid points in our spatial domain.
            target_vol (float): The total volume of the water droplet.
            args (dict): A dictionary containing all the parameters for the water droplet.
        '''
        super().__init__()
        self.args = args

        self.space_width = self.args['space_width']
        self.num_grids = self.args['num_grids']
        
        self.target_vol = self.args['volume']
        self.len_scale = self.target_vol ** (1/3)
        self.TCL_height = self.len_scale / 100

        self.model = Sequential([
            Input(shape=(2,)),
            Dense(64, activation = 'tanh'),
            Dense(64, activation = 'tanh'), 
            Dense(64, activation = 'tanh'),
            Dense(1, activation = 'linear')
        ])

        ## Initialize and assign our trainable Laplacian pressure.
        self.laplace_pressure = tf.Variable(0.1, dtype = tf.float32, trainable = True)
        self.laplace_pressure.assign(-(2.0 / np.sqrt(3)) / self.len_scale)

        ## Create variables that store coordinates and z-profiles at the edge of the droplet.
        ## Used for training hysteresis droplet.
        self.contact_mask_memory = tf.Variable(tf.zeros((self.num_grids ** 2, )), trainable = False, dtype = tf.float32)
        self.z_old_memory = tf.Variable(tf.zeros((self.num_grids ** 2, )), trainable = False, dtype = tf.float32)

    def call(self, inputs):
        '''
        Calls the model given a coordinate.

        Args:
            inputs (Tensor): A tensor of (x, y) coordinates.

        Returns:
            (Tensor): The respective z-profiles for each coordinate.
        '''
        return self.model(inputs)
    
    def warmup_model(self, weights_path = None, epochs = 3000, N_points = 10000):
        '''
        Initializes a 3D spherical cap shape before starting training.

        Args:
            weights_path (str): The path to the weights for initialized droplet.
            epochs (int): The number of iterations for the model to warmup.
            N_points (int): The number of coordinate samples.

        Returns:
            None
        '''
        if weights_path is not None:
            self.model = load_model(weights_path)

        else:
            optimizer_warmup = tf.keras.optimizers.Adam(learning_rate = 1e-4)

            print(f"Initializing Model: Epochs = {epochs}, Samples = {N_points}, Optimizer = {optimizer_warmup.get_config()['name']}")

            with tf.device('/GPU:0'):
                for _ in (pbar := trange(epochs, desc = "Epochs", ncols = 100)):
                    xy_sample = utils.get_random_coords(N_points)
                    with tf.GradientTape() as tape:
                        target_z = utils.get_spherical_cap(xy_sample)
                        z_pred = self.model(xy_sample)
                        warmup_loss = tf.reduce_mean(tf.square(z_pred[:, 0] - target_z))

                    ## Update the weights of the model only (not Laplacian pressure).
                    grads = tape.gradient(warmup_loss, self.model.trainable_variables)
                    optimizer_warmup.apply_gradients(zip(grads, self.model.trainable_variables))

                    pbar.set_postfix(loss = f'{warmup_loss:.4f}')
    
    def load_weights(self, model_weights, laplace_weights):
        '''
        Given the path to weights, load in pretrained model.

        Args:
            model_weights (str): A string representing the path to saved model.
            laplace_weights (str): A string representing the path to saved normalized pressure.

        Returns:
            None
        '''
        self.model = load_model(model_weights)
        self.laplace_pressure.assign(np.load(laplace_weights)['laplace_pressure'])

        z_pred = tf.reshape(self.call(utils.get_coords(self.space_width, self.num_grids)), [self.num_grids ** 2])
        contact_mask_now = tf.cast((z_pred >= 0.0) & (z_pred < self.TCL_height), tf.float32)

        self.contact_mask_memory.assign(contact_mask_now)
        self.z_old_memory.assign(z_pred)

    def save_weights(self, model_path, laplace_path):
        '''
        Given the path to weights, save the trained model and trainable variable.

        Args:
            model_path (str): A string representing the path that stores the model's weights.
            laplace_path (str): A string representing the path that stores the trainable variables.

        Returns:
            None
        '''
        self.model.save(model_path)
        np.savez(laplace_path, laplace_pressure = self.laplace_pressure.numpy())