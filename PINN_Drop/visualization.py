## This module contains visualization functions.

## Import necessary packages.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import PINN_Drop.utils as utils

import numpy as np
import matplotlib.pyplot as plt

def plot_visualizations(z_pred, Bo, epoch, metrics, space_width, num_grids):
    '''
    Plots the 3D simulated droplet, the PDE loss over epochs, and the maximum z-profile
    over epochs. Called from the 'train_pinn' function for visualization of each Bond number.

    Args:
        z_pred (Tensor): The tensor containing all the predicted z-profiles.
        Bo (float): The Bond number of the current droplet.
        epoch (int): The current epoch.
        metrics (dict): Dictionary containing all the metrics of the training session.
        space_width (int): The width of the grid.
        num_grids (int): Number of valid coordinates in the spatial domain.
    
    Returns:
        None
    '''
    fig = plt.figure(figsize = (12, 4))

    ax1 = fig.add_subplot(1, 3, 1, projection = '3d')
    plot_droplet(ax1, z_pred, space_width, num_grids, 
                 plot_title = f"Bo = {Bo} Droplet", x_label = "x", y_label = "y")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(range(epoch + 1), metrics['pde_loss'])
    ax2.set_title(f"Bo = {Bo} PDE Losses")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("PDE Loss")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(range(epoch + 1), metrics['z_profile'])
    ax3.set_title(f"Bo = {Bo} z-profiles")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("z-profiles")

    fig.tight_layout()
    plt.show()

def plot_yz_xz_views(z_pred, space_width, num_grids):
    '''
    Plots the 2D profile of the droplet when x = 0 and y = 0. 

    Args:
        z_pred: The predicted z-profiles of the droplet.
        space_width (float): The boundaries of the grid space.
        num_grids (int): The number of grid units.

    Returns:
        None
    '''
    x_full, y_full, z_y_slice, z_x_slice, z_max, z_min = utils.get_zx_zy_coords(z_pred, space_width, num_grids)

    fig, ax = plt.subplots(1, 2, figsize = (12, 4))

    ax[0].plot(y_full, z_y_slice, label = 'Slice at x=0')
    ax[0].set_title('Side View Along x-axis')
    ax[0].set_xlabel('y')
    ax[0].set_ylabel('Height z')
    ax[0].set_xlim(y_full[0], y_full[-1])
    ax[0].set_ylim(z_min, z_max * 1.05)
    ax[0].set_aspect('equal')
    ax[0].grid(True)

    ax[1].plot(x_full, z_x_slice, label = 'Slice at y=0')
    ax[1].set_title('Side View Along y-axis')
    ax[1].set_xlabel('x')
    ax[1].set_xlim(x_full[0], x_full[-1])
    ax[1].set_ylim(z_min, z_max * 1.05)
    ax[1].set_aspect('equal')
    ax[1].grid(True)

    fig.tight_layout()
    plt.show()

def plot_droplet(ax, z_pred, space_width, num_grids, plot_title, 
                 x_label, y_label, 
                 z_min = 0.0, z_max = 0.21, z_spacing = 0.1):
    '''
    Visualizes the simulated droplet figure.

    Args:
        ax (Axes3D): The plot that will contain the droplet.
        z_pred (Tensor): The predicted z-profile of the entire xy-grid.
        space_width (float): The boundaries of the grid space.
        num_grids (int): The number of grid units.
        plot_title (str): The title of the visualization plot.
        x_label (str): The x-axis label.
        y_label (str): The y-axis label.
        z_min (float): The minimum tick for the z-axis.
        z_max (float): The maximum tick for the z-axis.
        z_spacing (float): The spacing for the z-axis.

    Returns:
        None
    '''
    x_coords, y_coords, z_coords = utils.get_xyz_coords(z_pred, space_width, num_grids)

    ax.plot_surface(x_coords, y_coords, z_coords, cmap = 'viridis')

    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    z_range = z_coords.max() - z_coords.min()
    scaling_factor = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range / scaling_factor, y_range / scaling_factor, (z_range / scaling_factor) * 1.5])

    z_margin = z_spacing * (z_max - z_min)
    ax.set_zlim(z_min, z_max + z_margin)
    ax.set_xticks(np.linspace(x_coords.min(), x_coords.max(), 5))
    ax.set_yticks(np.linspace(y_coords.min(), y_coords.max(), 5))
    ax.set_zticks(np.linspace(z_min, z_max, 3))