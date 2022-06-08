from importlib.resources import path
from logging import root
from pathlib import Path
import matplotlib.pyplot as plt
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
    
    print("The graphs at epoch {}".format(i))
    with torch.no_grad():
        index = np.random.randint(0, getter.N_train)

        times = torch.linspace(0, final_time*dt, final_time, dtype=torch.float64).float()

        # print(dt)


        predicted_output, predicted_latent = model(getter.train_images[index, :2], times, dt)
        # print("out_shape", predicted_output.shape)
        # print(predicted_latent.shape)
        pca_encoded_trajectory = predicted_latent[:, :out_display].detach().numpy()
        # print("encoded", pca_encoded_trajectory.shape)
        # print(getter.train_positions[index].shape)
        # print("train images", getter.train_images[index, :-1, :out_display].shape)
        pca_train_trajectory = model.encode(getter.train_images[index, :-1, :out_display])
        # print("train",  pca_train_trajectory.shape)

        if pca_encoded_trajectory.shape[-1] > 2:

            # display in orange the predicted position and in blue the true position of the training set
            pca = PCA(n_components=2).fit(pca_train_trajectory)
            pca_encoded_trajectory = pca.transform(pca_encoded_trajectory)
            # print(getter.train_positions[index].shape)
            pca_train_trajectory = pca.transform(pca_train_trajectory)

        plt.figure(figsize=(15, 10))
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
        plt.plot(times, pca_train_trajectory[:,0], 'r', label="Ground truth Coord 1")
        plt.plot(times, pca_encoded_trajectory[:,0], 'orange', label="Predicted Coord 1")
        plt.xlabel("Time")
        plt.ylabel("First coord of PCA")
        plt.legend()
        # plt.show()

        if pca_encoded_trajectory.shape[-1] > 1:
            plt.subplot(2, 3, 3)
            plt.plot(times, pca_train_trajectory[:,1], 'r', label="Ground truth Coord 2")
            plt.plot(times, pca_encoded_trajectory[:,1], 'orange', label="Predicted Coord 2")
            plt.xlabel("Time")
            plt.ylabel("Second coord of PCA")
            plt.legend()
            # plt.show()

        index_img = np.random.randint(0, getter.train_images.shape[1]-1)
        plt.subplot(2, 3, 4)
        plt.imshow(getter.train_images[index, index_img, 0], cmap='gray')
        plt.subplot(2, 3, 6)
        plt.imshow(predicted_output[index_img, 0], cmap="gray")
        
        if root is None or name is None:
            plt.show()

        else:
            plt.savefig(Path(root) /f"{name}_epochs_{i}.png")
            plt.close()



def interactive_part_trajectory_image_plot(inputs_images, reconstructed_images, time_steps, dt):
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