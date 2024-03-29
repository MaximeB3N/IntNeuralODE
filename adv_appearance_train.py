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
from src.models.adversarial import FrameClassifier, SequenceClassifier, Discriminators

from src.utils.viz import plot_extrapolation
# Check how the finromation is contained inside the tqdm iterator

# Create the train function 
def evaluate(device, model, test_loader, loss_fn, psnr, ssim, tqdm_iterator):

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
            tqdm_iterator.set_description_str(f'Test {running_num_samples}/{len(test_loader)} Loss: {loss.item():.8f}')
            psnr.update(pred_images, batch_output_images)
            ssim.update(pred_images, batch_output_images)

    psnr_value = psnr.compute()
    ssim_value = ssim.compute()

    return running_loss/running_num_samples, psnr_value, ssim_value

def train(device, model, optimizer, discriminators, optimizer_discriminators, train_loader, loss_fn, adv_loss, tqdm_iterator):
    running_loss = 0.
    running_num_samples = 0

    for i, (batch_init_images, batch_times, batch_output_images) in enumerate(train_loader):
        batch_init_images = batch_init_images.to(device)
        # print(batch_times.shape)
        batch_times = batch_times[0].to(device)
        batch_output_images = batch_output_images.to(device)
        # compute the output of the model
        pred_images, pred_latent = model(batch_init_images, batch_times)
        
        # print("pred_images.shape", pred_images.shape)
        # print("pred_latent.shape", pred_latent.shape)
        # print("batch_output_images.shape", batch_output_images.shape)

        # compute the loss
        # print(out.shape, out.view(-1, batch_init_positions.shape[-1]).shape)
        # print(batch_true_positions.shape, batch_true_positions.view(-1, batch_init_positions.shape[-1]).shape)
        # print(out_images.shape, batch_true_images.shape)


        valid_frames = torch.ones(pred_images.shape[0]*pred_images.shape[1], 1).to(device)
        valid_sequence = torch.ones(pred_images.shape[0], 1).to(device)
        fake_frames = torch.zeros(pred_images.shape[0]*pred_images.shape[1], 1).to(device)
        fake_sequence = torch.zeros(pred_images.shape[0], 1).to(device)
        # -------------------
        # Train the Generator
        # -------------------
        loss = 0.
        loss += loss_fn(pred_latent, pred_images, batch_output_images)
        if i % 100 == 0:
            print("Recon loss", loss)
            print("seq loss")
            print(discriminators.forward_seq(pred_images)[:2], valid_sequence[:2])
            print(0.1 * adv_loss(discriminators.forward_seq(pred_images), valid_sequence))
            print("frame loss")
            print(discriminators.forward_frame(pred_images.view(-1, *pred_images.shape[2:]))[:2], valid_frames[:2])
            print(0.1 * adv_loss(discriminators.forward_frame(pred_images.view(-1, *pred_images.shape[2:])), valid_frames))

        loss += 0.1 * adv_loss(discriminators.forward_seq(pred_images).squeeze(), valid_sequence.squeeze())
        
        loss += 0.1 * adv_loss(discriminators.forward_frame(pred_images.view(-1, *pred_images.shape[2:])).squeeze(), valid_frames.squeeze())

        # .view(-1,batch_init_positions.shape[-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -----------------------
        # Train the Discriminator
        # -----------------------
        optimizer_discriminators.zero_grad()
        real_loss = adv_loss(discriminators.forward_seq(batch_output_images).squeeze(), valid_sequence.squeeze())
        real_loss += adv_loss(discriminators.forward_frame(batch_output_images.view(-1, *batch_output_images.shape[2:])).squeeze(), valid_frames.squeeze())
        fake_loss = adv_loss(discriminators.forward_seq(pred_images.detach()).squeeze(), fake_sequence.squeeze())
        fake_loss += adv_loss(discriminators.forward_frame(pred_images.view(-1, *pred_images.shape[2:]).detach()).squeeze(), fake_frames.squeeze())

        if i % 100 == 0:
            print("real loss", real_loss)
            print("fake loss", fake_loss)


        d_loss = 0.05 * (real_loss + fake_loss)
        d_loss.backward()#retain_graph=True)
        optimizer_discriminators.step()

        # update the progress bar
        tqdm_iterator.set_description_str(f'Train {running_num_samples}/{len(train_loader)} Loss: {loss.item():.8f}')
        running_loss += loss.item()
        running_num_samples += 1

    return running_loss/running_num_samples



def main(device, model, optimizer, discriminators, optimizer_discriminators, scheduler, epochs, train_loader, test_loader, 
    input_length, loss_fn, adv_loss, root_save_images, out_display=-1, checkpoint_image_interval=1, checkpoint_model_interval=10):
    
    device = model.device

    if out_display == -1:
        out_display = model.out_dim
    
    psnr = PSNR()
    ssim = SSIM(data_range=1.0)

    iterator = trange(1, epochs+1)
    # just for the plot part
    iterator_dict = {'train_loss': -1, 'test_loss': -1, 'PSNR': -1, 'SSIM': -1}
    iterator.set_postfix(iterator_dict)

    for i in iterator:
        # get a random time sample
        model.train()
        
        
        train_loss = train(device, model, optimizer, discriminators, optimizer_discriminators, train_loader, loss_fn, adv_loss, iterator)

        # display_results_fn(i, model, out_display, getter, getter.total_length, getter.dt)
        iterator_dict['train_loss'] = f"{train_loss:.8f}"
        iterator.set_postfix(iterator_dict)
        # loss_fn.forward_print(pred_latent, pred_images, batch_output_images)

        # evaluate the model
        if i % checkpoint_image_interval == 0:
            test_loss, psnr_value, ssim_value = evaluate(device, model, test_loader, loss_fn, psnr, ssim, iterator)

            iterator_dict['test_loss'] = f"{test_loss:.8f}"
            iterator_dict['PSNR'] = f"{psnr_value:.8f}"
            iterator_dict['SSIM'] = f"{ssim_value:.8f}"
            iterator.set_postfix(iterator_dict)
            iterator.update()
            
            print('\n', "-"*30)
            print(f"Epoch {i}/{epochs} Train Loss: {train_loss:.8f} Test Loss: {test_loss:.8f} PSNR: {psnr_value:.8f} SSIM: {ssim_value:.8f}")
            print("-"*30)
            
            display_one_trajectory(i, model, train_loader, test_loader, root_save_images, input_length)



        # save the model
        
        if i % checkpoint_model_interval == 0:
            print("Saving checkpoint...")
            torch.save(discriminators.state_dict(), f"{root_model}/discriminators_epoch_{i}.pt")
            torch.save(model.state_dict(), root_model / f'{checkpoint_name}_epoch_{i}.pt')
            print(f"Checkpoint '{root_model / f'{checkpoint_name}_epoch_{i}.pt'}' saved")
        

        # Update scheduler
        scheduler.step()
        loss_fn.step()

    return None

# Create the vizualization function
@torch.no_grad()
def display_one_trajectory(i, model, train_loader, test_loader, root_save_images, input_length):
    
    model.eval()
    
    # -------------------------------------------------- Train --------------------------------------------------
    # Modify this to use loader.sample() instead of running a for loop
    batch_init_images, batch_times, batch_output_images = next(iter(train_loader))
    
    batch_init_images = batch_init_images.to(model.device)
    batch_times = batch_times[0].to(device)
    batch_output_images = batch_output_images.to(model.device)[:,:, 0].unsqueeze(2)
    # compute the output of the model
    pred_images, _ = model(batch_init_images, batch_times)
    pred_images = pred_images[:,:, 0].unsqueeze(2)

    batch_init_images = batch_init_images[:, :input_length].unsqueeze(2)
    # print(pred_images.shape, batch_output_images.shape)
    # print(batch_init_images.shape)
    
    ground_truth_sequence = torch.cat([batch_init_images, batch_output_images], dim=1).cpu().numpy()
    prediction_sequence = torch.cat([batch_init_images, pred_images], dim=1).cpu().numpy()
    # print("ground_truth_sequence.shape", ground_truth_sequence.shape)
    # print("prediction_sequence.shape", prediction_sequence.shape)
    ground_truth_sequence = np.expand_dims(ground_truth_sequence[0], axis=0)
    prediction_sequence = np.expand_dims(prediction_sequence[0], axis=0)
    # print("ground_truth_sequence.shape", ground_truth_sequence.shape)
    # print("prediction_sequence.shape", prediction_sequence.shape)

    image_name = f"train_set_epoch_{i}_"
    plot_extrapolation(root_save_images, ground_truth_sequence, prediction_sequence, input_size=input_length, image_name=image_name, num_traj_plot=1)

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

    image_name = f"test_set_epoch_{i}_"
    plot_extrapolation(root_save_images, ground_truth_sequence, prediction_sequence, input_size=input_length, image_name=image_name, num_traj_plot=1)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/convnode_appearance64_adv.yaml")


    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    input_length = config["input_length"]
    target_length = config["target_length"]
    batch_size = config["batch_size"]

    print("-"*50 + "\n", "Loading dataset...")
    root = "/users/eleves-b/2019/maxime.bonnin/perso/PhyDNet/data/"
    
    train_dataset = NewMovingMNIST(root, is_train=True, n_frames_input=input_length, n_frames_output=target_length)
    test_dataset = NewMovingMNIST(root, is_train=False, n_frames_input=input_length, n_frames_output=target_length)

    # train_dataset = MovingMNIST(input_length=input_length, target_length=target_length, is_train=True, train_rate=0.8)
    # test_dataset = MovingMNIST(input_length=input_length, target_length=target_length, is_train=False, train_rate=0.8)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

    
    
    ode_out_dim = dim_dynamic
    ode_hidden_dim = config["ode_hidden_dim"]
    augment_dim = config["augment_dim"]

    print("-"*50 + "\n", "Creating ConvNODE with appearance model...")
    convnode = ConvNodeAppearance(device, encoder, decoder,
                                dim_dynamic, dim_appearance, input_length + 2, 
                                out_channels, ode_hidden_dim, ode_out_dim, augment_dim=augment_dim).to(device)

    print("Done.")

    print("-"*50 + "\n", "Creating discriminators ...")
    frame_classifier = FrameClassifier(device, 3).to(device)
    seq_classifier = FrameClassifier(device, 12).to(device)
    discriminators = Discriminators(seq_discriminator=seq_classifier, frame_discriminator=frame_classifier).to(device)
    optimizer_discriminators = torch.optim.Adam(discriminators.parameters(), lr=config["lr_discriminators"])
    print("Done.")
    
    # Create loss
    
    print("-"*50 + "\n", "Creating loss...")
    lr = config["lr"]
    reg_lambda = config["reg_lambda"]
    step_decay = config["step_decay"]
    decay_rate = config["decay_rate"]
    loss_fn = LatentRegularizerLoss(device, reg_lambda=reg_lambda)
    adv_loss = nn.BCELoss()
    print("Done.")

    # Create optimizer and scheduler
    print("-"*50 + "\n", "Creating optimizer...")
    
    optimizer = torch.optim.Adam(convnode.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_decay, decay_rate)
    print("Done.")

    # Train
    root_model = Path(config["root_model"])
    checkpoint_name = config["checkpoint_name"]
    checkpoint_image_interval = config["checkpoint_image_interval"]
    checkpoint_model_interval = config["checkpoint_model_interval"]
    root_images = Path(config["root_images"])
    epochs = config["epochs"]
    
    # convnode.load_state_dict(torch.load(os.path.join(root_model, "convnode_appearance64_epoch_100.pt")))

    assert train_dataset.dt == test_dataset.dt, "The dt must be the same for both dataset."
    
    print("-"*50 + "\n", "Training...")
    main(device, convnode, optimizer, discriminators, optimizer_discriminators, scheduler, epochs, train_loader, test_loader, 
    input_length, loss_fn, adv_loss, root_images, out_display=-1, checkpoint_image_interval=checkpoint_image_interval,
    checkpoint_model_interval=checkpoint_model_interval)
    print("Done.")
    