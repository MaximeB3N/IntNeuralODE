import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange
import argparse
from pathlib import Path

from src.utils.dataset import  MovingMNIST, NewMovingMNIST
from src.utils.loss import LatentRegularizerLoss
from src.utils.metrics import PSNR, SSIM

from src.models.resnet import ResNetEncoder64, ResNetDecoder64
from src.models.convnode import ConvNodeAppearance

from src.utils.viz import plot_extrapolation
# Check how the finromation is contained inside the tqdm iterator

# Create the train function 
def evaluate(device, model, test_loader, loss_fn, psnr, ssim):

    psnr.reset()
    ssim.reset()

    running_loss = 0.
    running_num_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_init_images, batch_times, batch_output_images in test_loader:
            batch_init_images = batch_init_images.to(device)
            batch_times = batch_times[0].to(device)
            batch_output_images = batch_output_images.to(device)
            # compute the output of the model
            pred_images, pred_latent = model(batch_init_images, batch_times)
            # compute the loss
            loss = loss_fn(pred_latent, pred_images, batch_output_images)
            # .view(-1,batch_init_positions.shape[-1])
            running_loss += loss.item()
            running_num_samples += 1
            # update the progress bar
            pred_images = pred_images[:,:, 0].unsqueeze(2)
            batch_output_images = batch_output_images[:,:, 0].unsqueeze(2)
            psnr.update(pred_images, batch_output_images)
            ssim.update(pred_images, batch_output_images)

    psnr_value = psnr.compute()
    ssim_value = ssim.compute()

    return running_loss/running_num_samples, psnr_value, ssim_value




def main(device, model, test_loader, 
    input_length, loss_fn, root_save_images, out_display=-1):
    
    device = model.device

    if out_display == -1:
        out_display = model.out_dim
    
    psnr = PSNR()
    ssim = SSIM(data_range=1.0)

    test_loss, psnr_value, ssim_value = evaluate(device, model, test_loader, loss_fn, psnr, ssim)

    print('Test loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f}'.format(test_loss, psnr_value, ssim_value))



    display_one_trajectory(model, test_loader, root_save_images, input_length)


    return None

# Create the vizualization function
@torch.no_grad()
def display_one_trajectory(model, test_loader, root_save_images, input_length):
    
    model.eval()

    # -------------------------------------------------- Test --------------------------------------------------

    batch_init_images, batch_times, batch_output_images = next(iter(test_loader))
    
    batch_init_images = batch_init_images.to(model.device)
    batch_times = batch_times[0].to(device)
    batch_output_images = batch_output_images.to(model.device)[:,:, 0].unsqueeze(2)
    # compute the output of the model
    pred_images, _ = model(batch_init_images, batch_times)
    pred_images = pred_images[:,:, 0].unsqueeze(2)

    batch_init_images = batch_init_images[:, :input_length].unsqueeze(2)
    
    ground_truth_sequence = torch.cat([batch_init_images, batch_output_images], dim=1).cpu().numpy()
    prediction_sequence = torch.cat([batch_init_images, pred_images], dim=1).cpu().numpy()
    # print("ground_truth_sequence.shape", ground_truth_sequence.shape)
    # print("prediction_sequence.shape", prediction_sequence.shape)
    ground_truth_sequence = np.expand_dims(ground_truth_sequence[0], axis=0)
    prediction_sequence = np.expand_dims(prediction_sequence[0], axis=0)
    # print("ground_truth_sequence.shape", ground_truth_sequence.shape)
    # print("prediction_sequence.shape", prediction_sequence.shape)

    image_name = f"test_set_epoch_evaluation_"
    plot_extrapolation(root_save_images, ground_truth_sequence, prediction_sequence, input_size=input_length, image_name=image_name, num_traj_plot=1)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/convnode_appearance64_eval.yaml")


    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    input_length = config["input_length"]
    target_length = config["target_length"]
    batch_size = config["batch_size"]

    print("-"*50 + "\n", "Loading dataset...")
    root = "/users/eleves-b/2019/maxime.bonnin/perso/PhyDNet/data/"
    
    Flag = False
    
    if input_length + target_length > 20:
        Flag = True
    test_dataset = NewMovingMNIST(root, is_train=Flag, n_frames_input=input_length, n_frames_output=target_length)

    # train_dataset = MovingMNIST(input_length=input_length, target_length=target_length, is_train=True, train_rate=0.8)
    # test_dataset = MovingMNIST(input_length=input_length, target_length=target_length, is_train=False, train_rate=0.8)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('Done.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    layers_encoder = config["layers_encoder"]
    input_length = config["input_length"]
    target_length = config["target_length"]
    dim_dynamic = config["dim_dynamic"]
    dim_appearance = config["dim_appearance"]
    out_channels = config["out_channels"]
    
    latent_encoder = 2 * dim_dynamic + dim_appearance
    latent_decoder = dim_dynamic + dim_appearance

    print("-"*50 + "\n", "Creating encoder...")
    encoder = ResNetEncoder64(layers_encoder, input_length + 2, latent_encoder).to(device)
    decoder = ResNetDecoder64(img_channel=out_channels, n_latent=latent_decoder).to(device)
    print("Done.")
    
    layers_encoder = config["layers_encoder"]
    ode_out_dim = dim_dynamic
    ode_hidden_dim = config["ode_hidden_dim"]
    augment_dim = config["augment_dim"]

    print("-"*50 + "\n", "Creating ConvNODE with appearance model...")
    convnode = ConvNodeAppearance(device, encoder, decoder, layers_encoder,
                                dim_dynamic, dim_appearance, input_length + 2, 
                                out_channels, ode_hidden_dim, ode_out_dim, augment_dim=augment_dim).to(device)

    print("Done.")
    
    # Create loss
    
    print("-"*50 + "\n", "Creating loss...")
    reg_lambda = config["reg_lambda"]
    step_decay = config["step_decay"]
    decay_rate = config["decay_rate"]
    loss_fn = LatentRegularizerLoss(device, reg_lambda=reg_lambda)
    print("Done.")


    # Train
    root_model = Path(config["root_model"])
    root_images = Path(config["root_images"])
    
    convnode.load_state_dict(torch.load(os.path.join(root_model, "convnode_appearance64_2000_epochs_final.pt")))
    
    print("-"*50 + "\n", "Evaluating...")
    main(device, convnode, test_loader, 
    input_length, loss_fn, root_images, out_display=-1)
    print("Done.")
    

