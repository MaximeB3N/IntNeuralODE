import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from ipywidgets import interact

import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


def plot_tsne_and_pca(tsne_encoded_trajectory, pca_encoded_trajectory, positions=None):
    '''
    Plot the t-SNE and PCA of the encoded trajectories (the transformed latent space vectors).
    '''
    fig = plt.figure(figsize=(20, 10))

    if not positions is None:
        ax = fig.add_subplot(133)
        ax.plot(positions[:, 0], positions[:, 1], '-')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Trajectory of the center of the ball')

    ax = fig.add_subplot(131)
    ax.plot(tsne_encoded_trajectory[:, 0], tsne_encoded_trajectory[:, 1], '-')
    ax.set_xlabel('Coord 1')
    ax.set_ylabel('Coord 2')
    ax.set_title('t-SNE viz of the encoded trajectory')

    ax = fig.add_subplot(132)
    ax.plot(pca_encoded_trajectory[:, 0], pca_encoded_trajectory[:, 1], '-')
    ax.set_xlabel('Coord 1')
    ax.set_ylabel('Coord 2')
    ax.set_title('PCA viz of the encoded trajectory')
    
    
    plt.show()

def plot_tsne_and_pca_portrait(times, tsne_encoded_trajectory, pca_encoded_trajectory, positions=None):
    '''
    Plot the t-SNE and PCA of the encoded trajectories (the transformed latent space vectors).
    '''
    fig = plt.figure(figsize=(20, 20))

    if not positions is None:
        ax = fig.add_subplot(133)
        ax.plot(times, positions[:, 0], '-')
        ax.set_xlabel('times')
        ax.set_ylabel('X axis')
        ax.set_title("X's Trajectory of the center of the ball")

        ax = fig.add_subplot(233)
        ax.plot(times, positions[:, 1], '-')
        ax.set_xlabel('times')
        ax.set_ylabel('Y axis')
        ax.set_title("Y's Trajectory of the center of the ball")


    ax = fig.add_subplot(131)
    ax.plot(times, tsne_encoded_trajectory[:, 0], '-')
    ax.set_xlabel('Times')
    ax.set_ylabel('Coord 1')
    ax.set_title("t-SNE viz of the X's encoded trajectory")

    ax = fig.add_subplot(231)
    ax.plot(times, tsne_encoded_trajectory[:, 1], '-')
    ax.set_xlabel('Times')
    ax.set_ylabel('Coord 2')
    ax.set_title("t-SNE viz of the Y's encoded trajectory")

    ax = fig.add_subplot(132)
    ax.plot(times, pca_encoded_trajectory[:, 0], '-')
    ax.set_xlabel('Times')
    ax.set_ylabel('Coord 1')
    ax.set_title("PCA viz of the X's encoded trajectory")

    ax = fig.add_subplot(232)
    ax.plot(times, pca_encoded_trajectory[:, 1], '-')
    ax.set_xlabel('Times')
    ax.set_ylabel('Coord 2')
    ax.set_title("PCA viz of the Y's encoded trajectory")
    
    
    plt.show()


def display_ode_trajectory(i, model, out_display, getter, final_time, dt):

    """
    Display the trajectory of the ODE model at epoch i, used when the model was only an ODE and inputs were the positions.

    Parameters
    ----------
    i : int, epoch number
    model : torch.nn.Module, the model to display
    out_display : int, number of coordinates to display
    getter : torch.utils.data.Dataset, the dataset to display
    final_time : int, the final time of the trajectory
    dt : float, the time step of the trajectory
    """
    print("The graphs at epoch {}".format(i))
    with torch.no_grad():
        index = np.random.randint(0, getter.N_train)

        times = torch.linspace(0., final_time*dt, final_time, dtype=torch.float64).float()

        y0, _, _ = getter.get_batch()

        predicted_output = model(getter.train_positions[index, 0].unsqueeze(0), times)

        pca_encoded_trajectory = predicted_output[:, -1, :out_display].detach().numpy()
        # print(getter.train_positions[index].shape)
        pca_train_trajectory = getter.train_positions[index, :, :out_display]

        if pca_encoded_trajectory.shape[-1] > 2:

            # display in orange the predicted position and in blue the true position of the training set
            pca = PCA(n_components=2).fit(pca_train_trajectory)
            pca_encoded_trajectory = pca.transform(pca_encoded_trajectory)
            # print(getter.train_positions[index].shape)
            pca_train_trajectory = pca.transform(pca_train_trajectory)


        if pca_encoded_trajectory.shape[-1] > 1:

            plt.plot(pca_encoded_trajectory[:,0], 
                    pca_encoded_trajectory[:,1], 'orange', label="Predicted")

            plt.plot(pca_train_trajectory[:,0], pca_train_trajectory[:,1], 'b', label="Ground truth")

            plt.xlabel("First coord")
            plt.ylabel("Second coord")
            plt.legend()
            plt.show()

       

        # print the X axis over the time
        plt.plot(times, pca_train_trajectory[:,0], 'r', label="Ground truth Coord 1")
        plt.plot(times, pca_encoded_trajectory[:,0], 'orange', label="Predicted Coord 1")
        plt.xlabel("Time")
        plt.ylabel("First coord of PCA")
        plt.legend()
        plt.show()

        if pca_encoded_trajectory.shape[-1] > 1:
            plt.plot(times, pca_train_trajectory[:,1], 'r', label="Ground truth Coord 2")
            plt.plot(times, pca_encoded_trajectory[:,1], 'orange', label="Predicted Coord 2")
            plt.xlabel("Time")
            plt.ylabel("Second coord of PCA")
            plt.legend()
            plt.show()
        

def display_convnode_trajectory(i, model, out_display, getter, final_time, dt, root=None, name=None):
    """
    Display the trajectory inside the latent space of the ConvNode model at epoch i and 
    comparing it with the trajectory of the true images encoded in the latent space.

    Parameters
    ----------
    i : int, epoch number
    model : torch.nn.Module, the model to display
    out_display : int, number of coordinates to display
    getter : torch.utils.data.Dataset, the dataset to display
    final_time : int, the final time of the trajectory
    dt : float, the time step of the trajectory
    root : str, the root directory to save the images
    name : str, the name of the image to save
    """
    device = model.device
    model.eval()
    print("The graphs at epoch {}".format(i))
    with torch.no_grad():
        index = np.random.randint(0, getter.N_train)

        times = torch.linspace(0, final_time*dt, final_time, dtype=torch.float64).float().to(device)

        # print(dt)


        predicted_output, predicted_latent = model(getter.train_images[index, :2].to(device), times, dt)
        # print("out_shape", predicted_output.shape)
        # print(predicted_latent.shape)
        pca_encoded_trajectory = predicted_latent[:, :out_display].cpu().detach().numpy()
        # print("encoded", pca_encoded_trajectory.shape)
        # print(getter.train_positions[index].shape)
        # print("train images", getter.train_images[index, :-1, :out_display].shape)
        pca_train_trajectory = model.encode(getter.train_images[index, :-1, :out_display].to(device)).cpu().detach().numpy()
        # print("train",  pca_train_trajectory.shape)

        if pca_encoded_trajectory.shape[-1] > 2:

            # display in orange the predicted position and in blue the true position of the training set
            pca = PCA(n_components=2).fit(pca_train_trajectory)
            pca_encoded_trajectory = pca.transform(pca_encoded_trajectory)
            # print(getter.train_positions[index].shape)
            pca_train_trajectory = pca.transform(pca_train_trajectory)

        fig = plt.figure(figsize=(15, 10))
        if pca_encoded_trajectory.shape[-1] > 1:
            plt.subplot(2, 3, 2)
            plt.plot(pca_encoded_trajectory[:,0], 
                    pca_encoded_trajectory[:,1], 'orange', label="Predicted")

            plt.plot(pca_train_trajectory[:,0], pca_train_trajectory[:,1], 'b', label="Ground truth")

            plt.xlabel("First coord")
            plt.ylabel("Second coord")
            plt.legend()
            # plt.show()
       

        # print the X axis over the time
        plt.subplot(2, 3, 1)
        plt.plot(times.cpu().numpy(), pca_train_trajectory[:,0], 'r', label="Ground truth Coord 1")
        plt.plot(times.cpu().numpy(), pca_encoded_trajectory[:,0], 'orange', label="Predicted Coord 1")
        plt.xlabel("Time")
        plt.ylabel("First coord of PCA")
        plt.legend()
        # plt.show()

        if pca_encoded_trajectory.shape[-1] > 1:
            plt.subplot(2, 3, 3)
            plt.plot(times.cpu().numpy(), pca_train_trajectory[:,1], 'r', label="Ground truth Coord 2")
            plt.plot(times.cpu().numpy(), pca_encoded_trajectory[:,1], 'orange', label="Predicted Coord 2")
            plt.xlabel("Time")
            plt.ylabel("Second coord of PCA")
            plt.legend()
            # plt.show()

        index_img = np.random.randint((getter.train_images.shape[1]-1)//2, getter.train_images.shape[1]-1)
        plt.subplot(2, 3, 4)
        plt.imshow(getter.train_images[index, index_img, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Image at time {index_img*dt:.3f}/{final_time*dt:.3f}")
        plt.subplot(2, 3, 6)
        plt.imshow(predicted_output[index_img, 0].cpu().detach().numpy(), cmap="gray")
        plt.title(f"Predicted image at time {index_img*dt:.3f}/{final_time*dt:.3f}")
        
        if root is None or name is None:
            plt.show()

        else:
            plt.savefig(Path(root) /f"{name}_epochs_{i}.png")
            fig.clf()
            plt.close()

def generate_interactive_plot(i, model, out_display, getter, final_time, dt, root=None, name=None):
    """
    Wrapper to create an interactive plot of the simulated images of the ConvNode model at epoch i. 
    It plots an image depending on the time slider position.

    Parameters
    ----------
    i : int, epoch number
    model : torch.nn.Module, the model to display
    out_display : int, number of coordinates to display
    getter : torch.utils.data.Dataset, the dataset to display
    final_time : int, the final time of the trajectory
    dt : float, the time step of the trajectory
    root : str, the root directory to save the images
    name : str, the name of the image to save (not used here but needed to be compatible with the other functions)
    """
    index = np.random.randint(0, getter.N_train)
    time_steps = np.linspace(0, final_time*dt, final_time)

    times = torch.arange(0, final_time*dt, dt)

    gd_images = getter.train_images[index, :-1]
    input_images = getter.train_images[index, :2]

    reconstructed_images, _ = model(input_images, times, dt)

    return interactive_part_trajectory_image_plot(gd_images, reconstructed_images, time_steps, dt)


def interactive_part_trajectory_image_plot(inputs_images, reconstructed_images, time_steps, dt):
    """
    Function to create an interactive plot of the simulated images of the ConvNode model at epoch i.

    Parameters
    ----------
    inputs_images : torch.Tensor, the input images of the model
    reconstructed_images : torch.Tensor, the predicted images of the model
    time_steps : np.array, the time steps of the trajectory
    dt : float, the time step of the trajectory
    """
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Input image", "Predicted image"))
    fig = go.FigureWidget(fig)
    # add a black background to the figure
    fig.add_image(z=inputs_images[0], row=1, col=1, name='true image')
    fig.add_image(z=reconstructed_images[0], row=1, col=2, name='predicted image')

    N_max_input = len(inputs_images)-1
    N_max_predicted = len(reconstructed_images)-1
    N_max = max(N_max_input, N_max_predicted)

    frac_input = 1. #N_max/N_max_predicted
    frac_predicted = 1. #N_max/N_max_input

    @interact(t=(time_steps.min(),time_steps.max(),dt))
    def update_plot(t):
        with fig.batch_update():
            # change the current point of 
            print(t/dt)
            print(int(frac_input*t/dt))
            print(int(frac_predicted*t/dt))
            fig.data[0].z = inputs_images[min(int(frac_input*t/dt), N_max_input)]
            fig.data[1].z = reconstructed_images[min(int(frac_predicted*t/dt), N_max_predicted)]

    return fig

def display_auto_encoder_reconstruction(model, dataset, N_samples_recon):

    """
    Display the reconstruction of the autoencoder model on the dataset.

    Parameters
    ----------
    model : torch.nn.Module, the model to display
    dataset : torch.utils.data.Dataset, the dataset to display
    N_samples_recon : int, the number of samples to display
    """
    # use the VAE to reconstruct images and there initial images
    N_samples_recon = 6
    plot_loader = DataLoader(dataset, batch_size=N_samples_recon)
    model.eval()
    with torch.no_grad():
        limit = np.random.randint(0, int(len(dataset)/N_samples_recon))
        for i, data in enumerate(plot_loader):
            if i <= limit:
                continue
            if len(data) == 2:
                input_image, _ = data
            else:
                # in that case data is the input image
                input_image = data

            input_image = input_image.float()
            img = model(input_image)
            img = img[:,0].cpu().detach().numpy()
            img = np.reshape(img, (N_samples_recon, 28, 28))
            input_image = input_image[:,0].cpu().detach().numpy()
            input_image = np.reshape(input_image, (N_samples_recon, 28, 28))
            height_plot = 5 
            width_plot = height_plot * int(np.ceil(N_samples_recon/2))
            fig, ax = plt.subplots(figsize=(width_plot,height_plot))
            for i in range(N_samples_recon):
                plt.subplot(2, N_samples_recon, i+1)
                plt.imshow(img[i], cmap='gray')
                plt.axis('off')
                
            for i in range(N_samples_recon):
                plt.subplot(2, N_samples_recon, i+1+N_samples_recon)
                plt.imshow(input_image[i], cmap='gray')
                plt.axis('off')
            plt.show()
            break


def plot_extrapolation(root_plot, ground_truth_sequence, prediction_sequence, input_size=10, num_traj_plot=1, image_name=None):
    """
    Plot the extrapolation of the model on the dataset by plotting the ground truth followed by the predictions
    (a red bar is added to separate the ground truth from the predictions).

    Parameters
    ----------
    root_plot : str, the root directory to save the images
    ground_truth_sequence : torch.Tensor, the ground truth sequence of the model
    prediction_sequence : torch.Tensor, the predicted sequence of the model
    input_size : int, the size of the input sequence
    num_traj_plot : int, the number of trajectories to plot
    image_name : str, the name of the image to save
    """

    # assert ground_truth_sequence.shape == prediction_sequence.shape
    assert (ground_truth_sequence.shape[0] == num_traj_plot and ground_truth_sequence.ndim == 5) or ground_truth_sequence.ndim == 4

    fcount = len(os.listdir(root_plot)) + 1

    if image_name is None:
        image_name = f"extrapolation_"


    num_plot = ground_truth_sequence.shape[0]
    gt_len_seq = ground_truth_sequence.shape[1]
    pred_len_seq = prediction_sequence.shape[1]
    max_len_seq = max(gt_len_seq, pred_len_seq)
    # min_len_seq = min(gt_len_seq, pred_len_seq)

    for traj_ind in range(num_plot):
        # Plot it with Matplotlib to compare ground truth and predictions
        fig_whole_seq, axes_whole_seq = plt.subplots(nrows=2, ncols=max_len_seq, figsize=(2*max_len_seq + 1.,4 + 2), squeeze=False)
        
        for i, ax in enumerate(axes_whole_seq.flat):

            if i < pred_len_seq:
                # The prediction part
                # ax = plt.subplot(2, len_seq, i+1+len_seq)
                ax.imshow(prediction_sequence[traj_ind, i % pred_len_seq].squeeze(), cmap='gray')
                # ax.axis('off')
                ax.set_title(f"frame={i+1}", fontsize="medium")
                ax.set_yticks([])
                ax.set_xticks([])
                
                if i == pred_len_seq:
                    ax.set_ylabel("Prediction")



            elif i - max_len_seq < gt_len_seq:
                # The ground truth part
                # ax = plt.subplot(2, len_seq, i+1)
                ax.imshow(ground_truth_sequence[traj_ind, (i - max_len_seq) % gt_len_seq].squeeze(), cmap='gray')
                
                # ax.axis('off')
                ax.set_yticks([])
                ax.set_xticks([])

                if i == 0:
                    ax.set_ylabel("Ground Truth")

            else:
                ax.axis('off')
                
            
            
        fig_whole_seq.tight_layout()

        r = fig_whole_seq.canvas.get_renderer()
        get_bbox = lambda axe: axe.get_tightbbox(r).transformed(fig_whole_seq.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, axes_whole_seq.flat)), mtrans.Bbox).reshape(axes_whole_seq.shape)

        xmax = np.array(list(map(lambda b: b.x1, bboxes.flat))).reshape(axes_whole_seq.shape)[:, input_size]
        xmin = np.array(list(map(lambda b: b.x0, bboxes.flat))).reshape(axes_whole_seq.shape)[:, input_size - 1]
        xs = np.c_[xmax[0], xmin[0]].mean(axis=1)[0]

        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat)))
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat)))
        ymax = ymax.max()
        ymin = ymin.min()
        # Draw a horizontal lines at those coordinates
        line = plt.Line2D([xs,xs],[ymin/2.,(1 + ymax)/2.], transform=fig_whole_seq.transFigure, color=(204./255, 0, 0), linestyle="--", linewidth=3.)
        # add text to the line
        text_input = plt.text(xs/2., 0.92, f"Input", transform=fig_whole_seq.transFigure, color=(204./255, 0, 0), fontsize=15, horizontalalignment='left')
        text_pred = plt.text((1+ xs)/2., 0.92, f"Prediction", transform=fig_whole_seq.transFigure, color=(204./255, 0, 0), fontsize=15, horizontalalignment='right')

        # fig_whole_seq.suptitle(f"Trajectory {traj_ind+1}")
        # print(text_input.get_bbox_patch().get_extents())

        fig_whole_seq.add_artist(line)
        fig_whole_seq.add_artist(text_input)
        fig_whole_seq.add_artist(text_pred)

        image_complete_name = image_name + f'comparison_id_{fcount}.png'

        # Save the figure
        fig_whole_seq.savefig(os.path.join(root_plot, image_complete_name))
        print("Created the image: ", image_complete_name)
        fcount += 1
        plt.close(fig_whole_seq)