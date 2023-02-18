# IntNeuralODE
This is the respository of my Master Thesis Project tackling the subject of "Continuous Motion Interpolation with Neural Differential Equations"

## Abstracts
Learning the physics from images is an appealing way to improve unsupervised video pre-
diction method. Knowing the underlying physics is not enough to describe images, we define
a deep neural network based on convolutions and neural ordinary differential equations that
disentangles the dynamics from the image content and learns the underlying dynamics. This is
the first ODE based framework to work on the disentanglement and allows appearance mod-
ification without altering the dynamics. Besides, we propose a model that is continuous-time
that can be also used for image interpolation.

## Structure
One can find the thesis report in the folder "report" and the code in the folder "src". There is also a folder call "notebook" where one can find the jupyter notebook used to explore before writing scripts to run the experiments. To better understand the architecture of the code, here is a tree representing the folder structure:


```
.
├── README.md
├── appearance_train.py : script to train a ConvNodeAppearance on the Moving MNIST
├── adv_appearance_train.py : same but with an adversarial manner
├── convnode_train.py : script to train a ConvNode on the Moving MNIST
├── appearance_train_different_lr.py : script to train a ConvNodeAppearance with different lr for AE and NODE
├── evaluate.py : script to evaluate the model on several test set (long term or real test set)


├── data
│   ├── MNIST : Used to generate the Moving MNIST training set
│   ├── MOVING_MNIST : Used to evaluate the model on the Moving MNIST test set

├── images : contains the images used in the report and more
│   ├── AE : Autoencoder
│   │   ├── Basic : results using 3 convolutions encoder (cf report for the details architecture) 
│   │   ├── ResNet : results using resnet34 encoder (cf report for the details architecture) 
│   ├── ODE : results of fitting an ODE for several toy examples
│   ├── AE_ODE : auto-encoder and Neural ODE together 
│   │   ├── Gravity : Gravity hole dataset images
│   │   ├── Spiral : Spiral dataset images
│   │   ├── Moving_MNIST : Moving MNIST images from experiments
│   ├── VAE

├── notebook : jupyter notebook used to explore the data and the models
│   ├── AE : Various notebooks to explore auto-encoder with various loss and models
│   ├── ODE : Exploration to look at the quality of the prediction and how well it generalizes
│   ├── ODE-AE : 
│   │   ├── AE_ODE_exploration.ipynb : First notebook to handle ConvODE training
│   │   ├── ConvODE_optimization_training.ipynb : Batch optimisation exploration
│   │   ├── ConvODE_together_exploration.ipynb : First try to train everything together
│   │   ├── ConvODE_disentangle.ipynb : First method to disentangle dynamics and appearance
│   │   ├── ConvODE_disentangle_different_encoder.ipynb : disentangling using different transformations between encoder and latent space
│   │   ├── metric_exploration.ipynb : test using SSIM and PSNR

├── report : contains the report of the project

├── config : several configs used to train models
│   ├── convnode_appearance64_adv.yaml : adversarial loss for convnode_appearance64
│   ├── convnode_appearance64_eval.yaml
│   ├── convnode_appearance64.yaml : basic convnode_appearance64
│   ├── convnode_appearance64_different_lr.yaml : convnode_appearance64 with lr different for the auto-encoder and the NODE

├── src : contains the code
│   ├── data : data generation code
│   │   ├── ball.py
│   │   ├── box.py
│   │   ├── generate.py

│   ├── models : all files related to the explored models
│   │   ├── adversarial.py : classifiers and discriminators for adversarial loss
│   │   ├── ae.py : basic convolutions auto-encoders
│   │   ├── anode.py : Augmented Neural Ordinary Equations implementations
│   │   ├── cnnae.py : Deeper convolutions auto-encoders
│   │   ├── convnode.py : To create an auto-encoder along with a Neural ODE w/ or w/out disentangling
│   │   ├── node.py : Neural ODE implementation 
│   │   ├── resnet.py : ResNet auto-encoders
│   │   ├── vae.py : VAE implementations using ResNet or simple conv as backbone
│   ├── utils
│   │   ├── dataset.py : dataset handling for NODE and MovingMNIST dataset generator
│   │   ├── loss.py : various losses (MSE, Regularization, Perceptual loss, ...)
│   │   ├── metrics.py : PSNR and SSIM implementations
│   │   ├── utils.py : ball_shape, spatial encoding, ...
│   │   ├── viz.py : plot functions

├── sim : simulation to vizualize how the trajectories are generated (not for training, viz purpose only)
│   ├── background.png : background image of the simulation (not resized)
│   ├── frame.png : same but resized
│   ├── bouncing_main.py : script to make the image "shark.png" to move on the background "frame.png"
│   ├── gaussian_main.py : script to move a gaussian ball on a black background
```



