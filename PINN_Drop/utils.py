## This module contains any utility or helper functions.

## Import necessary packages.
import tensorflow as tf
import numpy as np
from scipy.optimize import fsolve

def get_coords(space_width = 2.0, num_grids = 160):
    '''
    Given the dimensions of the spatial domain, return all the (x, y) coordinates the grid.

    Args:
        space_width (float): The width of our grid.
        num_grids (int): The number of grid points in the spatial domain.

    Returns:
        xy_full_tensor (Tensor): A tensor containing all the (x, y) coordinate pairs.
    '''
    x_full = np.linspace(-space_width / 2, space_width / 2, num_grids)
    y_full = np.linspace(-space_width / 2, space_width / 2, num_grids)

    x_grid, y_grid = np.meshgrid(x_full, y_full)

    xy_full = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    xy_full_tensor = tf.convert_to_tensor(xy_full, dtype = tf.float32)

    return xy_full_tensor

def get_random_coords(N, space_width = 2.0):
    '''
    Samples random coordinates from our grid.
    
    WARNING: Cannot be used in hysterisis case.

    Args:
        N (int): The number of samples we want to choose.
        space_width (float): The width of our grid (has to be a square grid).
    
    Returns:
        xy_sample_tensor (Tensor): A tensor containing random (x, y) coordinates from our spatial grid.
    '''
    xy_sample_tensor = tf.random.uniform((N, 2), minval = -space_width / 2, maxval = space_width / 2, dtype = tf.float32)
    return xy_sample_tensor

def get_spherical_cap(xy_coords):
    '''
    Initializes a water droplet shape (base case).

    Args:
        xy_coords (Tensor): A tensor containing (x, y) coordinates.

    Returns:
        (Tensor): A tensor containing all the respective (x, y) coordinates.
    '''
    x = xy_coords[:, 0]
    y = xy_coords[:, 1]

    r2 = tf.square(x) + tf.square(y)

    ## Assign initial values to parameters.
    R = tf.constant(0.7, dtype = tf.float32)
    H0 = tf.constant(0.2, dtype = tf.float32)

    h = H0 * (1 - (r2 / tf.square(R)))

    return tf.maximum(h, -0.5)

def get_pinn_args(space_width = 2.0, num_grids = 160, volume = 0.1, density = None, gravity = None, 
                  st = None, Bo = 0, k_mask = 320, high_angle = None, low_angle = None, 
                  tilt_angle = 0, surface_pos = [], width = [0.01], surface_angles = [60], 
                  weight_list = [100, 1000, 1000, 0, 100, 0]):
        '''
        Used to define the numerical and physics parameters for the PINN model.

        Args:
            space_width (float): The numerical range of the grid which the droplet lies on.
            num_grids (int): The number of grid points along a single axis.
            volume (float): The unitless volume of the droplet.
            density (float): The density of the liquid droplet.
            gravity (float): The gravity of the environment.
            st (float): The surface tension of the liquid.
            Bo (float): The Bond number of the liquid (if none, Bo is calculated using the above parameters).
            k_mask (float): Sharpness of the contact edge mask. 
            high_angle (float): The angle of the advancing side of the droplet (when hysteresis is applied).
            low_angle (float): The angle of the receding side of the droplet (when hysteresis is applied).
            tilt_angle (float): The angle of the surface tilt (when hysteresis is applied).
            surface_pos (list): The x-positions of unique surfaces (multiple contact angles).
            width (list): The steepness of the change of the contact angles from surface to surface.
            surface_angles (list): A list of all the contact angles for each surface.
            weight_list (list): The training weights for each loss term (PDE Loss, Volume Loss, Angle Loss, Motion Loss, Center Loss, y-center Loss).
        Returns:
            (dict): The args dictionary.
        '''
        return {'space_width': space_width, 
                'num_grids': num_grids, 
                'volume': volume, 
                'density': density,
                'gravity': gravity, 
                'st': st, 
                'Bo': Bo,
                'k_mask': k_mask,
                'high_angle': tf.constant(np.tan(np.radians(high_angle)), dtype = tf.float32) if high_angle is not None else None, 
                'low_angle': tf.constant(np.tan(np.radians(low_angle)), dtype = tf.float32) if low_angle is not None else None, 
                'tilt_angle': tilt_angle, 
                'surface_pos': surface_pos, 
                'width': width, 
                'surface_angles': surface_angles, 
                'weight_list': weight_list}

def compute_bond_num(p, g, len_scale, st):
    '''
    Calculates Bond Number given parameters.

    Args:
        p (float): The density of the liquid.
        g (float): The gravity of the current space.
        len_scale (float): The length scale of the liquid (1/3 of the liquid's volume).
        st (float): The surface tension of the liquid.

    Returns:
        Bo (float): The Bond Number of the given liquid.
    '''
    Bo = (p * g * (len_scale ** 2)) / st
    return Bo

def get_multiple_target_slope(xy_tensor, boundaries, angles, k):
    '''
    Given multiple surfaces, return each target slope.

    Args:
        xy_tensor (Tensor): A tensor representing all the (x, y) coordinates in spatial domain.
        boundaries (List): A list of all the x-positions of unique surfaces.
        angles (List): A list of all the contact angles for each surface.
        k (List): The steepness of the change of the contact angles from surface to surface.

    Returns:
        target_slopes (Tensor): A tensor containing all the target slopes across the grid.
    '''
    boundaries_tf = tf.constant(boundaries, dtype = tf.float32)
    angles_tf = tf.constant(angles, dtype = tf.float32)

    x_pos = xy_tensor[:, 0]
    x_pos = tf.expand_dims(x_pos, 0)

    sigmoid_steps = tf.sigmoid((10.0 / tf.expand_dims(k, 1)) * (x_pos - tf.expand_dims(boundaries_tf, 1)))

    dtheta = angles_tf[1:] - angles_tf[:-1]
    theta_degs = angles_tf[0] + tf.reduce_sum(tf.expand_dims(dtheta, 1) * sigmoid_steps, axis = 0)

    target_slopes = tf.tan(tf.constant(np.pi / 180.0, dtype = tf.float32) * theta_degs)

    return target_slopes

def get_grads(tape, z_pred, xy_tensor):
    '''
    Computes the necessary gradients of the z-profile.

    Args:
        tape (GradientTape): Tensorflow's gradient function that keeps track of changes.
        z_pred (Tensor): A tensor for the predicted z-profiles.
        xy_tensor (Tensor): A tensor storing all the (x, y) coordinates.

    Returns:
        (List): A list of all the gradients.
    ''' 
    ## Calculate all the first order gradients.
    dz_dx = tape.gradient(z_pred, xy_tensor)[:, 0]
    dz_dy = tape.gradient(z_pred, xy_tensor)[:, 1]

    ## Calculate all the second order and partial derivatives.
    d2z_dx2 = tape.gradient(dz_dx, xy_tensor)[:, 0]
    d2z_dy2 = tape.gradient(dz_dy, xy_tensor)[:, 1]
    d2z_dxdy = tape.gradient(dz_dy, xy_tensor)[:, 0]

    return [dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy]

def mae(pred, gt):
    '''
    Calculates the MAE between prediction and ground truth.

    Args:
        pred (Tensor): Predicted values.
        gt (Tensor): Ground truth values.

    Returns:
        (float): The MAE between predicted and ground truth values.
    '''
    return np.abs(pred - gt)

def height_equation(h, theta, vol):
    '''
    Calculates the height of the droplet.

    Args:
        h (float): The height of the spherical cap.
        theta (float): The contact angle of the cap.
        vol (float): The total volume of the cap.

    Returns:
        (float): If the parameters are correct, function should return zero.
    '''
    return (2 * (h ** 3)) + ((h ** 3) * np.cos(theta)) - (((3 * vol) / np.pi) * (1 - np.cos(theta)))

def get_height(theta_rad, vol):
    '''
    Find the height given the contact angle and the volume of the droplet under zero gravity.
    Acts as the ground truth height for Bond number 0.

    Args:
        theta_rad (float): The contact angle of the droplet.
        vol (float): The total volume of the droplet.

    Returns:
        (float): The analytical height of the droplet.
    '''
    h_sol = fsolve(height_equation, 1.0, args = (theta_rad, vol))
    return h_sol[0]

def get_curve(theta, vol):
    '''
    Finds the curvature of an analytical spherical cap (under zero gravity).

    Args:
        theta (float): The contact angle of the droplet.
        vol (float): The total volme of the droplet.

    Returns:
        (float): The analytical curvature of the droplet under zero gravity.
    '''
    theta_rad = theta * (np.pi / 180.0)
    height = get_height(theta_rad, vol)
    R = height / (1 - np.cos(theta_rad))
    return 2 / R

def get_xyz_coords(z_pred, space_width, num_grids, epsilon = 1e-4):
    '''
    Returns x, y, and z coordinates from model prediction. Used for visualization 
    purposes.

    Args:
        z_pred (Tensor): The predicted z-profiles of the droplet.
        space_width (float): The boundaries of the grid space.
        num_grids (int): The number of grid units.

    Returns:
        x_grid (np.array): The x-coordinates of the grid.
        y_grid (np.array): The y-coordinates of the grid.
        z_masked (np.array): The z-coordinates of the droplet.
    '''
    x_full = np.linspace(-space_width / 2, space_width / 2, num_grids)
    y_full = np.linspace(-space_width / 2, space_width / 2, num_grids)

    x_grid, y_grid = np.meshgrid(x_full, y_full)

    ## From the predicted z-profile, set z-values to zero if they're less than the epsilon threshold.
    ## This is done because the model returns predicted z-profiles for all the xy-coordinates in our 
    ## grid. However, only z-values greater than zero are valid.
    z_full = np.reshape(z_pred.ravel(), (num_grids, num_grids))
    z_masked = np.ma.masked_where(z_full < epsilon, z_full)

    return x_grid, y_grid, z_masked

def get_zx_zy_coords(z_pred, space_width, num_grids):
    '''
    Returns xy coordinates, the yz slice where x = 0, the xz slice where y = 0, and 
    the minimum and maximum of the z-values.

    Args:
        z_pred (Tensor): The predicted z-profiles of the droplet.
        space_width (float): The boundaries of the grid space.
        num_grids (int): The number of grid units.

    Returns:
        x_full (np.array): The x-coordinates.
        y_full (np.array): The y-coordinates.
        z_y_slice (np.array): The z-coordinates where x = 0.
        z_x_slice (np.array): The z-coordinates where y = 0.
        z_max (float): The maximum value of the z-profile.
        z_min (float): The minimum value of the z-profile (set to 0).
    '''
    x_full = np.linspace(-space_width / 2, space_width / 2, num_grids)
    y_full = np.linspace(-space_width / 2, space_width / 2, num_grids)
    z_full = np.reshape(z_pred, (num_grids, num_grids))

    z_y_slice = np.maximum(0.0, z_full[:, num_grids // 2])
    z_x_slice = np.maximum(0.0, z_full[num_grids // 2, :])

    z_max = max(np.max(z_y_slice), np.max(z_x_slice))
    z_min = 0.0

    return x_full, y_full, z_y_slice, z_x_slice, z_max, z_min