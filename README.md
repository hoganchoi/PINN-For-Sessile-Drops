# Modeling Sessile Drop Behavior Using Physics-Informed Neural Networks

## Setting Up Conda and Python Environment
Please create and activate your Conda environment using the code below.

```markdown
conda create --name [name-of-your-virtual-environment] python=3.10
conda activate [name-of-your-virtual-environment]
```
(NOTE: Python 3.10 was used in order to install `Tensorflow=2.10.0`, which has GPU compatibility)

If you have a Nvidia GPU, please download `conda-forge`, `cudatoolkit`, and `cudnn`. The following command was used on our personal computer.

```markdown
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Please activate the Conda environment and download all the required packages from `requirements.txt`.

```markdown
pip install -r requirements.txt
```

## Model Weights and Python Notebooks
We uploaded three Python notebooks to our ``demo`` folder. Each notebook goes over the workflow we did to obtain the figures from our paper. The following are:
 - ``pinn_demo_1.ipynb``: Standard training session and produces Fig 2.
 - ``pinn_demo_2.ipynb``: Training session for multiple contact angles and produces Fig 1.
 - ``pinn_demo_3.ipynb``: Hysteresis training session and produces Fig 6.
   
We also provided saved model weights to our ``saved_models`` folder. We organized the folder as shown below.
 - ``benchmark``: Contains the neural network and trainable variable weights of the benchmark cases outlined in Fig 4.
 - ``nine_cases``: Contains the neural network and trainable variable weights of the starting seed for the nine hysteresis cases in Fig 6.
 - ``PINN_hysteresis_model_Bo_2_9.keras``: The weights of the neural network for the hysteresis droplet (Fig 6).
 - ``PINN_hysteresis_state_Bo_2_9.npz``: The weights of the trainable normalized Laplace variable for the hysteresis droplet.
 - ``PINN_model_Bo_2_9.keras``: The weights of the neural network of droplet of Bond number 2.9 formed on flat surface. The starting seed for hysteresis training.
 - ``PINN_state_Bo_2_9.npz``: The weights of the trainable variable for a droplet formed on flat surface with Bond number 2.9.

The notebooks, by default, trains the droplet formations from scratch. However, to load in pre-trained weights, please do the following.
```python
model.load_weights(model_weights = "Destination to neural network weights. Is a .keras and denoted as 'model'.",
                   laplace_weights = "Destination to trainable normalized Laplacian weights. Is a .npz and denoted as 'state'.")
```
If you want to save trained model weights, please do the following.
```python
model.save_weights(model_path = "Save path for neural network weights.",
                   laplace_path = "Save path for normalized Laplacian weights.")
```
For more information, please check the ``pinn_model.py`` module in the ``PINN_Drop`` folder.
