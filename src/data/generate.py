import numpy as np
import torch
from tqdm import trange

from src.utils.utils import gaussian_density

# Generation of the dataset of uniform gaussian ball over all the space 
def create_gaussian_dataset(r_min, r_max, n_samples, size, margin=1., n_balls=1):
    """ 
    Create a dataset of gaussian balls randomly placed in the image

    Parameters
    ----------
    r_min : float, minimum radius of the gaussian ball
    r_max : float, maximum radius of the gaussian ball
    n_samples : int, number of samples to generate
    size : int, size of the image
    margin : float, margin to leave between the ball and the border of the image
    n_balls : int, number of balls to generate in the image
    """
    samples = []
    indices_matrix = np.array([[i,j] for i in range(size) for j in range(size)])
    # print(indices_matrix)
    # print(indices_matrix.shape)
    eps = 1E-5

    for i in range(n_samples):
        image = np.zeros((1, size, size))
        for _ in range(n_balls):
            sigma = np.random.uniform(r_min, r_max+eps)
            # create a gaussian ball
            mu = np.random.uniform(low=margin, high=size - 1 - margin, size=(2))
            # mu = np.random.randint(size, size=(2))
            # compute the density over the image and normalize it
            single_image = gaussian_density(indices_matrix, mu, sigma).numpy().copy()
            image += single_image.reshape(1, size, size)
        # standardization of the image
        image = (image - image.min()) / (image.max() - image.min())
        # image = (image - image.mean()) / (image.std() + eps)
        # image = (image - image.min()) / (image.max() - image.min()) 
        samples.append([image.reshape(1, size, size), np.array([mu[0]/(size ), mu[1]/(size)])])
        
    return samples


def generate_gravity_hole_ball_positions_and_velocity(box, N, N_frames, dt, infos):
    """
    Generate positions and velocities of a gravity hole ball.

    Parameters
    ----------
    box : Box object
    N : int, number of samples to generate
    N_frames : int, number of frames to generate for each sample
    dt : float, time step
    infos : dict, contains all the information about the ball

    Comments :
        infos example : 
        infos = {
        "MARGIN_MIN":5,
        "MIN_INIT_VELOCITY":200.,
        "WIDTH":28,
        "HEIGHT":28,
        "RADIUS":3
    }
    """
    dataset = []
    
    for _ in trange(N):
        x, y = np.random.uniform(infos["MARGIN_MIN"], infos["WIDTH"] - 1 - infos["MARGIN_MIN"], 2)

        vx, vy = np.random.uniform(-infos["MIN_INIT_VELOCITY"], infos["MIN_INIT_VELOCITY"], 2)

        box.reset(x0=x, y0=y, vx0=vx, vy0=vy)
        trajectory = []

        x, y = box.position
        vx, vy = box.velocity
        modified_position = ( np.array([x, y]).copy() - np.array([infos["WIDTH"]/2., infos["HEIGHT"]/2.]))/np.array([infos["WIDTH"], infos["HEIGHT"]])
        modified_velocity = np.array([vx, vy])/(infos["WIDTH"]*infos["HEIGHT"])
        trajectory.append(np.concatenate((modified_position, modified_velocity), axis=0))

        for _ in range(N_frames - 1):
            x, y = box.move(dt)
            vx, vy = box.velocity
            modified_position = ( np.array([x, y]).copy() - np.array([infos["WIDTH"]/2., infos["HEIGHT"]/2.]))/np.array([infos["WIDTH"], infos["HEIGHT"]])
            modified_velocity = np.array([vx, vy])/(infos["WIDTH"]*infos["HEIGHT"])
            trajectory.append(np.concatenate((modified_position, modified_velocity), axis=0))

        dataset.append(trajectory)

    return np.array(dataset)

def generate_gravity_hole_ball_images(box, N, N_frames, dt, infos, background_image=None):
    """
    Generate images of a gravity hole ball with the possibility to use a background image.
    
    Parameters
    ----------
    box : Box object
    N : int, number of samples to generate
    N_frames : int, number of frames to generate for each sample
    dt : float, time step
    infos : dict, contains all the information about the ball
    background_image : np.array, background image to use

    Comments :
        infos example : 
        infos = {
        "MARGIN_MIN":5,
        "MIN_INIT_VELOCITY":200.,
        "WIDTH":28,
        "HEIGHT":28,
        "RADIUS":3
    }
    """

    dataset = []
    INDICES_MATRIX = np.array([[i,j] for i in range(infos["WIDTH"]) for j in range(infos["HEIGHT"])])

    for _ in trange(N):
        x, y = np.random.uniform(infos["MARGIN_MIN"], infos["WIDTH"] - 1 - infos["MARGIN_MIN"], 2)

        vx, vy = np.random.uniform(-infos["MIN_INIT_VELOCITY"], infos["MIN_INIT_VELOCITY"], 2)

        box.reset(x0=x, y0=y, vx0=vx, vy0=vy)
        
        trajectory = []

        x, y = box.move(0.)
        ball_img = gaussian_density(INDICES_MATRIX, np.array([x, y]), infos["RADIUS"]).reshape(infos["WIDTH"], infos["HEIGHT"]).numpy()

        if background_image is not None:
                ball_img = (ball_img - ball_img.min())/(ball_img.max() - ball_img.min())
                ball_img = np.where(ball_img > 0.15, ball_img, 0.5*background_image)
                # ball_img = 0.5*background_image + 0.5*ball_img
                
        ball_img = (ball_img - ball_img.min())/(ball_img.max() - ball_img.min())
        trajectory.append(ball_img)
        for _ in range(N_frames - 1):
            x, y = box.move(dt)
            ball_img = gaussian_density(INDICES_MATRIX, np.array([x, y]), infos["RADIUS"]).reshape(infos["WIDTH"], infos["HEIGHT"]).numpy()
            # normalize it
            if background_image is not None:
                ball_img = (ball_img - ball_img.min())/(ball_img.max() - ball_img.min())
                ball_img = np.where(ball_img > 0.15, ball_img, 0.5*background_image)
                # ball_img = 0.5*background_image + 0.5*ball_img
            ball_img = (ball_img - ball_img.min())/(ball_img.max() - ball_img.min())
            trajectory.append(ball_img.copy())

        dataset.append(trajectory)

    return np.array(dataset)


def add_average_velocity(trajectories, Num_pos_velocity, dt):
    """
    Concatenate the average velocity of the last Num_pos_velocity positions to the trajectories. 
    Used in order to have all the information of the dynamics.

    Parameters
    ----------
    trajectories : torch.Tensor, shape (N, N_frames, 2), the trajectories of the particles
    Num_pos_velocity : int, number of positions used to compute the average velocity
    dt : float, time step

    """
    i = 0

    N_frames = trajectories.shape[1]
    n = N_frames - Num_pos_velocity
    velocity = (trajectories[:, i + 1: n + i + 1] - trajectories[:, i : n + i])/dt

    for i in range(1, Num_pos_velocity):
        velocity += (trajectories[:, i + 1: n + i + 1] - trajectories[:, i : n + i])/dt

    trajectories = torch.cat((torch.clone(trajectories[:, :-Num_pos_velocity]), velocity/Num_pos_velocity), dim=-1)
    
    return trajectories