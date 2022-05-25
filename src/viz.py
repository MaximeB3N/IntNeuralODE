import matplotlib.pyplot as plt
import numpy as np




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