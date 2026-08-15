## This module contains functions related to training the model.

## Import necessary packages.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PINN_Drop.utils as utils
import PINN_Drop.losses as losses
import PINN_Drop.eval as eval
from PINN_Drop.visualization import plot_visualizations

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def train_pinn(pinn_model, Bo_steps, per_Bo_step, final_Bo_step, lr = 1e-4, eval_model = False, 
                    show_visual = False, use_monte = False, n_samples = 10000):
    '''
    This is the training code for simulating various droplets on a flat surface.

    Args:
        pinn_model (PINNDrop): The neural network predicting the z-profiles of the droplet.
        Bo_steps (int): The number of warmup steps needed to reach the target Bond number.
        per_Bo_step (int): The number of epochs per warmup step.
        final_Bo_step (int): The number of epochs for the target Bond number.
        lr (int): The initial learning rate (default is 1e-4).
        eval_model (bool): Whether to keep track of performance metrics.
        show_visual (bool): Visualize droplet formation after each Bond number step.
        use_monte (bool): Whether to use the Monte-Carlo method during training.
        n_samples (int): If using Monte-Carlo, the number of random coordinates to sample.

    Returns:
        metrics (dict): If evaluation mode is true, then returns a dictionary with all the stored metrics.
    '''
    if not use_monte:
        xy_full_tensor = utils.get_coords(pinn_model.space_width, pinn_model.num_grids)

    ## Calculate the Bond number of the liquid if Bond number isn't specified.
    target_Bo = pinn_model.args['Bo']
    if target_Bo is None:
        target_Bo = utils.compute_bond_num(pinn_model.args['density'], pinn_model.args['gravity'], pinn_model.len_scale, pinn_model.args['st'])

    Bo = tf.constant(target_Bo, dtype = tf.float32)

    if Bo_steps > 0:
        Bo_step = (target_Bo / Bo_steps)
        temp_Bo = tf.constant(0, dtype = tf.float32)
    else:
        Bo_step = 0
        temp_Bo = Bo

    total_epochs = (Bo_steps * per_Bo_step) + final_Bo_step
    track_steps = Bo_steps

    if eval_model or show_visual:
        metrics = {'total_loss': [], 'pde_loss': [], 'z_profile': [], 'angle_accuracy': []}

    if len(pinn_model.args['surface_pos']) == 0:
        target_slope = tf.constant(np.tan(np.radians(pinn_model.args['surface_angles'][0])), dtype = tf.float32)
    else:
        target_slope = utils.get_multiple_target_slope(xy_full_tensor, pinn_model.args['surface_pos'], 
                                                        pinn_model.args['surface_angles'], pinn_model.args['width'])

    pinn_model.compile(optimizer = Adam(learning_rate = lr))

    with tf.device('/GPU:0'):
        for epoch in range(total_epochs):
            with tf.GradientTape(persistent = True) as tape:
                ## If training using the monte-carlo method, sample random xy-coordinates for every epoch.
                if use_monte:
                    xy_full_tensor = utils.get_random_coords(n_samples, space_width = pinn_model.space_width)

                tape.watch(xy_full_tensor)

                z_pred = pinn_model(xy_full_tensor, training = True)
                z_pred = tf.reshape(z_pred, (-1, ))

                grad_list = utils.get_grads(tape, z_pred, xy_full_tensor)
                dz_dx, dz_dy, *_ = grad_list
                grad_mag = tf.sqrt((dz_dx ** 2) + (dz_dy ** 2) + 1e-10)

                mask_domain = tf.sigmoid(pinn_model.args['k_mask'] * (z_pred))
                mask_edge = tf.sigmoid(pinn_model.args['k_mask'] * (z_pred)) - tf.sigmoid(pinn_model.args['k_mask'] * (z_pred - pinn_model.TCL_height))

                ## Because this is training on a flat surface, the 'sin_alpha' parameter is set to 0.
                pde_loss = losses.pde_loss(xy_full_tensor, z_pred, mask_domain, 0, pinn_model.laplace_pressure, 
                                                      temp_Bo, pinn_model.len_scale, grad_list)

                vol_loss = losses.volume_loss(pinn_model.target_vol, pinn_model.space_width, pinn_model.num_grids, 
                                            mask_domain, z_pred, use_monte = use_monte)
                
                angle_loss = losses.angle_loss(target_slope, mask_edge, grad_mag, 0, 
                                            pinn_model.args)
                
                ## Create dummy value for motion and y loss.
                motion_loss = 0
                y_loss = 0
                center_loss = losses.center_loss(xy_full_tensor, z_pred, mask_domain)

                total_loss = ((pinn_model.args['weight_list'][0] * pde_loss) + (pinn_model.args['weight_list'][1] * vol_loss)
                            + (pinn_model.args['weight_list'][2] * angle_loss) + (pinn_model.args['weight_list'][3] * motion_loss) 
                            + (pinn_model.args['weight_list'][4] * center_loss) + (pinn_model.args['weight_list'][2] * y_loss))

                grads = tape.gradient(total_loss, pinn_model.trainable_variables)
                pinn_model.optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))

            ## Calculate and store the metrics if eval mode is on.
            if eval_model or show_visual:
                metrics['total_loss'].append(total_loss)
                metrics['pde_loss'].append(pde_loss)
                metrics['z_profile'].append(tf.reduce_max(z_pred))
                angle_mae = eval.angle_eval(z_pred, grad_mag, pinn_model.args['surface_angles'][0])
                metrics['angle_accuracy'].append(angle_mae)

            ## Print changing variables for tracking purposes.
            print(
                "Training Epoch: {:01d} / {:01d} |  Bond Number: {:.3f}  |  " \
                "PDE Loss: {:.3f}".format(epoch, total_epochs, temp_Bo.numpy(), pde_loss), end = '\r'
                )

            if track_steps > 0 and ((epoch + 1) % per_Bo_step) == 0:
                ## Visualize the current simulated droplet, the PDE loss graph, and the z-profile plot.
                if show_visual:
                    print()
                    
                    pred = pinn_model.predict(utils.get_coords(pinn_model.space_width, pinn_model.num_grids), verbose = 0)
                    plot_visualizations(pred, temp_Bo, epoch, metrics, pinn_model.space_width, pinn_model.num_grids)
                    
                temp_Bo = temp_Bo + Bo_step
                track_steps = track_steps - 1

        print()
        
        ## If eval_model, return metrics.
        if eval_model:
            return metrics

def train_hysteresis_pinn(pinn_model, lr = 1e-4, total_epochs = 15000, eval_model = False, update_threshold = 1e-3):
    '''
    This is the modified training code for simulating droplets on a tilted surface.

    Args:
        pinn_model (PINNDrop): The neural network predicting the z-profiles of the droplet.
        lr (int): The initial learning rate (default is 1e-4).
        total_epochs (int): The number of training epochs.
        eval_model (bool): Whether to keep track of performance metrics.
        update_threshold (float): The threshold that determines whether the parameters have spiked.

    Returns:
        metrics (dict): If evaluation mode is true, then returns a dictionary with all the stored metrics.
    '''
    xy_full_tensor = utils.get_coords(pinn_model.space_width, pinn_model.num_grids)

    if pinn_model.args['Bo'] is None:
        Bo = utils.compute_bond_num(pinn_model.args['density'], pinn_model.args['gravity'], pinn_model.len_scale, pinn_model.args['st'])
        Bo = tf.constant(Bo, dtype = tf.float32)
    else:
        Bo = tf.constant(pinn_model.args['Bo'], dtype = tf.float32)

    if eval_model:
        metrics = {'total_loss': [], 'pde_loss': [], 'z_profile': [], 'angle_loss': [], 'volume_loss': []}

    ## Compute tilt of surface (tilt angle is 0 if surface is flat).
    tilt_rad = np.radians(pinn_model.args['tilt_angle'])
    sin_alpha = tf.constant(np.sin(tilt_rad), dtype = tf.float32)

    target_slope = (pinn_model.args['high_angle'] + pinn_model.args['low_angle']) / 2

    pinn_model.compile(optimizer = Adam(learning_rate = lr))

    epoch = 0
    ## This variable is used for tracking gradient spikes. This value is set to 0 when
    ## a spike occurs and the optimizer is reset.
    track_epoch = 0

    with tf.device('/GPU:0'):
        for epoch in range(total_epochs):
            with tf.GradientTape(persistent = True) as tape:
                ## Store previous weight parameters to determine if there was a spike or not.
                old_vars = [tf.identity(v) for v in pinn_model.trainable_variables]

                tape.watch(xy_full_tensor)

                z_pred = pinn_model(xy_full_tensor, training = True)
                z_pred = tf.reshape(z_pred, (-1, ))

                grad_list = utils.get_grads(tape, z_pred, xy_full_tensor)
                dz_dx, dz_dy, *_ = grad_list
                grad_mag = tf.sqrt((dz_dx ** 2) + (dz_dy ** 2) + 1e-10)

                mask_domain = tf.sigmoid(pinn_model.args['k_mask'] * z_pred)
                mask_edge = tf.sigmoid(pinn_model.args['k_mask'] * (z_pred)) - tf.sigmoid(pinn_model.args['k_mask'] * (z_pred - pinn_model.TCL_height))

                pde_loss = losses.pde_loss(xy_full_tensor, z_pred, mask_domain, sin_alpha, 
                                           pinn_model.laplace_pressure, Bo, pinn_model.len_scale, grad_list)

                vol_loss = losses.volume_loss(pinn_model.target_vol, pinn_model.space_width, pinn_model.num_grids, 
                                              mask_domain, z_pred)
                
                angle_loss = losses.angle_loss(target_slope, mask_edge, grad_mag, sin_alpha, 
                                               pinn_model.args)
                
                motion_loss, contact_mask_new, z_new = losses.motion_loss(pinn_model.contact_mask_memory, pinn_model.z_old_memory, z_pred, 
                                                                          grad_mag, pinn_model.TCL_height, pinn_model.args)
                
                pinn_model.contact_mask_memory.assign(contact_mask_new)
                pinn_model.z_old_memory.assign(z_new)

                ## Create dummy value for center loss.
                center_loss = 0

                y_loss = losses.y_center_loss(xy_full_tensor, mask_domain, z_pred)

                total_loss = ((pinn_model.args['weight_list'][0] * pde_loss) + (pinn_model.args['weight_list'][1] * vol_loss)
                            + (pinn_model.args['weight_list'][2] * angle_loss) + (pinn_model.args['weight_list'][3] * motion_loss) 
                            + (pinn_model.args['weight_list'][4] * center_loss) + (pinn_model.args['weight_list'][5] * y_loss))

                grads = tape.gradient(total_loss, pinn_model.trainable_variables)
                pinn_model.optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))

                ## Determine the global norm of the change of parameter weights.
                updates = [
                    v - v_old
                    for v, v_old in zip(pinn_model.trainable_variables, old_vars)
                ]
                update_norm = tf.linalg.global_norm(updates)

                ## If the global norm exceeds the 'update_threshold', use the previous parameter weights and
                ## reset the optimizer with 1/10 of the current learning rate. Reset the tracking epoch to 0.
                if update_norm > update_threshold:
                    for v, v_old in zip(pinn_model.trainable_variables, old_vars):
                        v.assign(v_old)
                    pinn_model.optimizer = Adam(learning_rate = lr / 10)
                    track_epoch = 0

            ## Calculate and store the metrics if eval mode is on.
            if eval_model:
                metrics['total_loss'].append(total_loss)
                metrics['pde_loss'].append(pde_loss)
                metrics['z_profile'].append(tf.reduce_max(z_pred))
                metrics['angle_loss'].append(angle_loss)
                metrics['volume_loss'].append(vol_loss)

            track_epoch = track_epoch + 1

            ## Print changing variables for tracking purposes.
            print(
                "Training Epoch: {:01d}  |  Track Epoch: {:01d}  |  Bond Number: {:.3f}  |  " \
                "PDE Loss: {:.3f}  |  Tilt Angle: {:.3f}".format(epoch, track_epoch, Bo.numpy(), pde_loss, sin_alpha.numpy()), end = '\r'
                )

        ## If eval_model, return metrics.
        if eval_model:
            return metrics