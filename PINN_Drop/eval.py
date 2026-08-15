## This module contains functions related to evaluating the model.

## Import necessary packages.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PINN_Drop.utils as utils

import numpy as np

## Finds the Mean Absolute Error between the predicted angle and the true angle.
def angle_eval(z_pred, grad_mag, gt_angle):
    ## Keep track of the angles formed on the edge of the droplet.
    z_sorted = np.sort(z_pred)
    z_argsorted = np.argsort(z_pred)

    ## Pick the 10 smallest z-profiles that are greater than 0. These are the edge points of the 
    ## droplet.
    edge_min_indicies = z_argsorted[np.where(z_sorted >= 0.0)[0][:10]]
    edge_grads = np.array(grad_mag)[edge_min_indicies]

    ## Get the average gradients of the 10 smallest z-profiles.
    edge_grads_avg = np.mean(edge_grads).item() 
    edge_angles = np.arctan(edge_grads_avg) * (180.0 / np.pi)

    ## Compute the MAE between the predicted angle and the true angle.
    angle_rmse = utils.mae(edge_angles, gt_angle)

    ## Return the MAE value between the predicted and true contact angle.
    return angle_rmse

def curve_eval(curve_pred, gt_curve):
    curve_error = utils.mae(curve_pred, gt_curve)
    return curve_error