## This module contains functions related to evaluating the model.

## Import necessary packages.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PINN_Drop.utils as utils

import numpy as np

## Finds the Mean Absolute Error between the predicted angle and the true angle.
def angle_eval(z_pred, grad_mag, gt_angle):
    '''
    Finds the Mean Absolute Error between the predicted angle and the true angle.

    Args:
        z_pred (Tensor): The predicted z-profiles from the PINN model.
        grad_mag (Tensor): The predicted slope of the droplet's surface.
        gt_angle (float): The target slope of the droplet.

    Returns:
        angle_rmse (float): The Root Mean Squared Error between the predicted and ground truth slope.
    '''
    z_sorted = np.sort(z_pred)
    z_argsorted = np.argsort(z_pred)

    ## Pick the 10 smallest z-profiles that are greater than 0. These are the edge points of the 
    ## droplet.
    edge_min_indicies = z_argsorted[np.where(z_sorted >= 0.0)[0][:10]]
    edge_grads = np.array(grad_mag)[edge_min_indicies]

    edge_grads_avg = np.mean(edge_grads).item() 
    edge_angles = np.arctan(edge_grads_avg) * (180.0 / np.pi)

    angle_rmse = utils.mae(edge_angles, gt_angle)

    return angle_rmse

def curve_eval(curve_pred, gt_curve):
    '''
    Finds the Mean Absolute Error between the predicted curve and the true curve.
    Used only when the Bond number is zero.

    Args:
        curve_pred (float): The predicted curvature from the PINN model.
        gt_curve (float): The true curvature of the droplet

    Returns:
        curve_error (float): The MAE between the predicted and ground truth curvature.
    '''
    curve_error = utils.mae(curve_pred, gt_curve)
    return curve_error