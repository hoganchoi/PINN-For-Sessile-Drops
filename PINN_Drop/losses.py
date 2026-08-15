## This module contains all the loss functions for training the PINN model.

## Import necessary packages.
import tensorflow as tf

def pde_loss(xy_tensor, z, mask_domain, sin_alpha, laplace_pressure, 
             Bo, len_scale, grad_list):
    '''
    Computes the Laplace pressure loss by finding the difference between the 
    pressure outside and inside the droplet.

    Args:
        xy_tensor (Tensor): The tensor containing all the (x, y) coordinates in the spatial domain.
        z (Tensor): Tensor containing z-profiles at each (x, y) coordinate.
        mask_domain (Tensor): The mask distinguishing the center area of the droplet.
        sin_alpha (float): The tilt angle of the surface.
        laplace_pressure (Variable): The trainable parameter representing delta Laplacian pressure normalized by surface tension.
        Bo (float): The Bond number of the liquid.
        len_scale (float): The length scale in relation to the total volume of the drop.
        grad_list (List): A list containing all the necessary gradients of the z-profile.

    Returns:
        (Tensor): A tensor containing the normalize PDE loss.
        (float): The predicted curvature of the droplet.
    '''
    dz_dx, dz_dy, d2z_dx2, d2z_dy2, d2z_dxdy = grad_list

    denom = (1 + (dz_dx ** 2) + (dz_dy ** 2)) ** (3 / 2)
    numer = ((1 + (dz_dy ** 2)) * d2z_dx2) - (2 * dz_dx * dz_dy * d2z_dxdy) + ((1 + (dz_dx ** 2)) * d2z_dy2)
    curvature = numer / (denom + 1e-10)

    x = xy_tensor[:, 0]
    hydrostatic_term = (z * ((1 - (sin_alpha ** 2)) ** (1/2))) + (x * sin_alpha)

    laplace_residual = curvature - (laplace_pressure + (hydrostatic_term * (Bo / (len_scale ** 2))))

    pde_loss = tf.reduce_sum(mask_domain * tf.square(laplace_residual)) / (tf.reduce_sum(mask_domain) + 1e-10)
    laplace_mean = tf.reduce_sum(mask_domain * (numer / (denom + 1e-10))) / (tf.reduce_sum(mask_domain) + 1e-6)

    return pde_loss / (tf.square(laplace_mean) + 1e-10)

def volume_loss(target_vol, space_width, num_grids, mask_domain, z, use_monte = False):
    '''
    Calculate the volume loss of the droplet.

    Args:
        target_vol (float): The ground truth value for the volume of the droplet.
        space_width (float): The width of the grid.
        num_grids (float): The number of grid points in the spatial domain.
        mask_domain (Tensor): A mask isolating the center area of the droplet.
        z (Tensor): The z-profiles of every (x, y) coordinates.
        use_monte (bool): Whether training session uses the Monte-Carlo method.

    Returns:
        (Tensor): The normalized loss of the volume.
    '''
    if use_monte:
        inside_mask = z > 0
        z_inside = tf.boolean_mask(z, inside_mask)

        domain_area = tf.constant(space_width ** 2, dtype = tf.float32)

        N_total = tf.cast(tf.shape(z)[0], tf.float32)
        N_inside = tf.cast(tf.shape(z_inside)[0], tf.float32)
        area_base = domain_area * (N_inside / N_total)

        vol = area_base * tf.reduce_mean(z_inside)

    else:
        dx = space_width / num_grids
        dy = space_width / num_grids

        cell_area = dx * dy

        vol = tf.reduce_sum(mask_domain * z) * cell_area

    vol_loss = tf.square(vol - target_vol)

    return vol_loss / tf.square(target_vol)

def angle_loss(target_slope, mask_edge, grad_mag, sin_alpha, pinn_args):
    '''
    Calculate the angle loss of the droplet.

    Args:
        target_slope (Tensor): The ground truth slope of the droplet.
        mask_edge (Tensor): A mask isolating the edges of the droplet.
        grad_mag (Tensor): The predicted slope of the droplet's surface.
        sin_alpha (float): The tilt angle of the surface.
        pinn_args (dict): A dictionary containing physical parameters of the environment.

    Returns:
        angle_loss (Tensor): The normalized loss of the contact angle.
    '''
    if sin_alpha == 0:
        angle_loss = tf.reduce_sum((mask_edge * tf.square((grad_mag - target_slope) / target_slope))
                                   /
                                   (tf.reduce_sum(mask_edge) + 1e-10))
    else:
        angle_loss = tf.reduce_sum((mask_edge * tf.square(tf.nn.relu(grad_mag - pinn_args['high_angle']))) +
                (mask_edge * tf.square(tf.nn.relu(pinn_args['low_angle'] - grad_mag)))) / (tf.reduce_sum(mask_edge) + 1e-10)
        angle_loss = angle_loss / tf.square(target_slope)

    return angle_loss

def motion_loss(contact_mask_memory, z_old_memory, z, grad_mag, TCL_height, pinn_args):
    '''
    Penalize physically impossible droplet movements during hysteresis training.

    Args:
        contact_mask_memory (Tensor): Contains the area where the old droplet makes contact with the surface.
        z_old_memory (Tensor): Contains the z-profile of the droplet previous to update.
        z (Tensor): Contains the new predicted z-profile of the droplet.
        grad_mag (Tensor): The contact slopes of the droplet.
        TCL_height (float): The threshold for determining the contact edges of the droplet.
        pinn_args (dict): A dictionary containing physical parameters of the environment.

    Returns:
        normalize_direction_loss (Tensor): The normalized hysterisis loss.
        contact_mask_new (Tensor): The contact area of the new droplet.
        z_new (Tensor): The new z-profile of the droplet after removing invalid z-profiles.
    '''
    contact_mask_now = tf.cast((z >= 0.0) & (z < TCL_height), tf.float32)
    moved_mask = tf.abs(contact_mask_memory - contact_mask_now)

    new_points = contact_mask_now * (1 - contact_mask_memory)
    old_points = contact_mask_memory * (1 - contact_mask_now)

    v_CL = z - z_old_memory

    ## Depending on the movement taken place, determine which moved due to the advancing slope (positive v_CL)
    ## and which moved due to the receding slope (negative v_CL). The masks obtained from the following are
    ## valid motions that follow physical laws.
    advancing_mask = tf.cast((grad_mag > pinn_args['high_angle']) & (v_CL > 0), tf.float32)
    receding_mask = tf.cast((grad_mag < pinn_args['low_angle']) & (v_CL < 0), tf.float32)
    allowed_motion = advancing_mask + receding_mask

    advancing_attempt_mask = tf.cast((v_CL > 0), tf.float32)
    receding_attempt_mask = tf.cast((v_CL < 0), tf.float32)
    penalty_adv_mask = new_points * (advancing_attempt_mask - advancing_mask)
    penalty_rec_mask = old_points * (receding_attempt_mask - receding_mask)
    
    ## If there are any movements of the contact mask that aren't included in the 'allowed_motion' set, 
    ## these are considered to be artifacts. Store these coordinates in 'direction_penalty_mask' set.
    direction_penalty_mask = moved_mask * (penalty_adv_mask + penalty_rec_mask)
    w_z = tf.clip_by_value(1.0 - z / TCL_height, 0.0, 1.0)

    direction_loss = (tf.reduce_sum(direction_penalty_mask * w_z * tf.square(v_CL)) 
                      / 
                      (tf.reduce_sum(direction_penalty_mask) + 1e-6))
    normalized_direction_loss = direction_loss / tf.square(TCL_height / 1)

    valid_new_points = new_points * allowed_motion
    valid_old_points = old_points * allowed_motion
    contact_mask_new = contact_mask_memory + valid_new_points - valid_old_points

    direction_penalty = tf.cast(direction_penalty_mask, tf.bool)
    z_new = tf.where(direction_penalty, z_old_memory, z)

    return normalized_direction_loss, contact_mask_new, z_new

def y_center_loss(xy_tensor, mask_domain, z):
    '''
    Loss term that ensures droplet doesn't move away from the y-center when training 
    under hysterisis.

    Args:
        xy_tensor (Tensor): Containing all the xy-coordinates in our grid.
        mask_domain (Tensor): The coordinates of the center of the droplet.
        z (Tensor): Predicted z-profiles for each xy-coordinate.

    Returns:
        y_loss (Tensor): The center of mass of the droplet w.r.t. the center along the y-axis.
    '''
    y = xy_tensor[:, 1]
    mass = mask_domain * z

    total_mass = tf.reduce_sum(mass) + 1e-6
    y_com = tf.reduce_sum(mass * y) / total_mass

    y_loss = tf.square(y_com)

    return y_loss
    
def center_loss(xy_tensor, z, mask_domain):
    '''
    Loss term that penalizes unnecessary movements of the droplet during training. Used
    during training droplets with a surface tilt of zero.

    Args:
        xy_tensor (Tensor): Containing all the xy-coordinates of the grid.
        z (Tensor): Predicted z-profiles of each xy-coordinate.
        mask_domain (Tensor): The coordinates of the center of the droplet.

    Returns:
        center_loss (Tensor): The center of mass of the droplet w.r.t. the center of the grid.
    '''
    x = xy_tensor[:, 0]
    y = xy_tensor[:, 1]

    mass = mask_domain * z
    total_mass = tf.reduce_sum(mass) + 1e-10

    com_x = tf.reduce_sum(mass * x) / total_mass
    com_y = tf.reduce_sum(mass * y) / total_mass

    center_loss = tf.square(com_x) + tf.square(com_y)

    return center_loss
