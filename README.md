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