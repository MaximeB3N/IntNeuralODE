import torch
import numpy as np
from sklearn.model_selection import train_test_split
import gzip
import os
import random
import torch.utils.data as data

# from .viz import display_ode_trajectory
from .utils import add_spatial_encoding, create_spatial_encoding

    
class BatchGetterMultiImages:
    def __init__(self, batch_time, batch_size, n_stack, total_length, dt, images, frac_train):
        # N: number of trajectories
        # M: number of time steps
        # D: dimension of the state space
        # positions: (N, T, D)
        self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        if isinstance(images, torch.Tensor):
            self.true_images = images.float()

        elif isinstance(images, np.ndarray):
            self.true_images = torch.from_numpy(images).float()

        else:
            assert False, "positions must be either a torch.Tensor or a np.ndarray"

        self.N_train = int(images.shape[0]*frac_train)

        self.train_times = self.times #[:self.N_train]
        self.test_times = self.times #[self.N_train:]
        self.train_images = self.true_images[:self.N_train]
        self.test_images = self.true_images[self.N_train:]
        self.batch_size = batch_size
        self.n_stack = n_stack
        self.batch_time = batch_time
        self.dt = dt
        self.total_length = total_length

    
    def get_batch(self):
        index = np.random.randint(0, self.N_train, self.batch_size)
        s = torch.from_numpy(np.random.choice(np.arange(self.train_times.shape[0] - self.batch_time, dtype=np.int64), 1, replace=False))
        batch_y0 = self.train_images[index, s:s+self.n_stack+1].squeeze(0) # (M, D)
        batch_t = self.train_times[:self.batch_time]  # (T)
        batch_y = torch.stack([self.train_images[index, s + i] for i in range(self.batch_time)], dim=1).squeeze(1)  # (T, M, D)
        print(s)
        return batch_y0, batch_t, batch_y


# Own Moving MNIST dataset (using only test set)        
# ----------------------------------------------------------------------------------------------------------------------
class MovingMNIST:
    def __init__(self, input_length=10, target_length=10, is_train=True, train_rate=0.8, path_data="data/mnist_test_seq.npy"):
        assert target_length + input_length <= 20, f"The dataset has only 20 frames per sequence and not {target_length + input_length}"

        self.input_length = input_length
        self.target_length = target_length
        self.data_length = input_length + target_length
        self.dt = 1.0/20.
        self.path_data = path_data
        self.is_train = is_train

        dataset = np.load(path_data).swapaxes(0, 1)[:]
        dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        dataset = train_test_split(dataset, train_size=train_rate, random_state=42)

        self.spatial_encoding_x, self.spatial_encoding_y = create_spatial_encoding(dataset[0].shape[-1])
        self.spatial_encoding_x = torch.from_numpy(self.spatial_encoding_x).unsqueeze(0)
        self.spatial_encoding_y = torch.from_numpy(self.spatial_encoding_y).unsqueeze(0)
        
        self.batch_time = torch.linspace(self.dt, (self.target_length)*self.dt, self.target_length).float()

        if is_train:
            self.train_data = torch.from_numpy(dataset[0]).float()
            print("Shape of the dataset: ", self.train_data.shape)
        else:
            self.test_data = torch.from_numpy(dataset[1]).float()
            print("Shape of the dataset: ", self.test_data.shape)

    def __len__(self):
        if self.is_train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            # data is of shape (data_length, 1, 64, 64)
            data = self.train_data[index, :self.data_length]
            
        else:
            # data is of shape (data_length, 1, 64, 64)
            data = self.test_data[index, :self.data_length]
        
        # Transfrom the input to be of shape (len_seq, input_length, 64, 64)
        input_data = data[:self.input_length]
        # print("input data 1 shape: ", input_data.shape)
        # Transfrom the input to be of shape (len_seq, target_length + 2, 64, 64) with spatial encoding
        input_data = torch.cat([input_data, self.spatial_encoding_x, self.spatial_encoding_y], dim=0).float()
        # print("input data 1 shape: ", input_data.shape)
        
        target_data = data[self.input_length:self.input_length+self.target_length].float().unsqueeze(1)
        # Add the spatial encoding to the target
        target_data = torch.from_numpy(add_spatial_encoding(target_data.cpu().numpy())).float()
        # print("target data shape: ", target_data.shape)
        # print(self.batch_time.shape)

        # target_data = torch.cat([target_data, self.spatial_encoding_x, self.spatial_encoding_y], dim=1).float()
        # print("target data shape: ", target_data.shape)
        return input_data, self.batch_time, target_data

# Moving MNIST dataset from PhyDNet code        
# ----------------------------------------------------------------------------------------------------------------------

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root, is_train):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class NewMovingMNIST(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, spatial_depth=0, num_objects=[2], transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(NewMovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                root_mov_mnist = "data/MOVING_MNIST/"
                self.dataset = load_fixed_set(root_mov_mnist, False)
                
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.spatial_depth = spatial_depth
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1
        self.dt = self.step_length_

        self.batch_time = torch.linspace(self.step_length_, (self.n_frames_output)*self.step_length_, self.n_frames_output).float()
        self.spatial_encodings = create_spatial_encoding(self.image_size_, spatial_depth)
        self.spatial_encodings = torch.from_numpy(np.array(self.spatial_encodings)).unsqueeze(1)


    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        # if self.transform is not None:
        #     images = self.transform(images)

        r = 1 # patch size (a 4 dans les PredRNN)
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()

        # print("input data 1 shape: ", input_data.shape)
        # Transfrom the input to be of shape (len_seq, target_length + 2, 64, 64) with spatial encoding
        # print("input data shape: ", input.shape, "spatial encoding shape: ", self.spatial_encoding_x.shape, self.spatial_encoding_y.shape)
        input = torch.cat([input.squeeze(), *self.spatial_encodings], dim=0).float()
        # print("input data 1 shape: ", input_data.shape)
        # Add the spatial encoding to the target
        # output = torch.from_numpy(add_spatial_encoding(output.cpu().numpy(), depth=0)).float()
        # print()
        # print(input.size())
        # print(output.size())

        return input, self.batch_time, output

    def __len__(self):
        return self.length

# Old dataset (not used anymore but used to train NeuralODE when there is a lot of data)        
# ----------------------------------------------------------------------------------------------------------------------


# class batchGetterPositions:
#     def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train, noise=-1):
#         self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
        
#         if isinstance(positions, torch.Tensor):
#             self.true_positions = positions.float()

#         elif isinstance(positions, np.ndarray):
#             self.true_positions = torch.from_numpy(positions).float()

#         else:
#             assert False, "positions must be either a torch.Tensor or a np.ndarray"

#         self.noise = noise
#         self.N_train = int(positions.shape[0]*frac_train)
#         if noise > 0 and noise < 1:
#             # adding gaussian noise to the true positions
#             self.true_positions = self.true_positions + torch.normal(0, noise, size=self.true_positions.shape)
#             self.true_positions = self.true_positions.float()

#         self.train_times = self.times[:self.N_train]
#         self.test_times = self.times[self.N_train:]
#         self.train_positions = self.true_positions[:self.N_train]
#         self.test_positions = self.true_positions[self.N_train:]
#         self.n_samples = n_samples
#         self.batch_time = batch_time
#         self.dt = dt
#         self.total_length = total_length

#     def get_batch(self):
#         s = torch.from_numpy(np.random.choice(np.arange(self.N_train - self.batch_time, dtype=np.int64), self.n_samples, replace=False))
#         batch_y0 = self.train_positions[s]  # (M, D)
#         batch_t = self.train_times[:self.batch_time]  # (T)
#         batch_y = torch.stack([self.train_positions[s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
#         return batch_y0, batch_t, batch_y

# class BatchGetterMultiTrajectories:
#     def __init__(self, batch_time, n_samples, total_length, dt, positions, frac_train):
#         # N: number of trajectories
#         # M: number of time steps
#         # D: dimension of the state space
#         # positions: (N, T, D)
#         self.times = torch.linspace(0., total_length*dt, total_length, dtype=torch.float64).float()
#         if isinstance(positions, torch.Tensor):
#             self.true_positions = positions.float()

#         elif isinstance(positions, np.ndarray):
#             self.true_positions = torch.from_numpy(positions).float()

#         else:
#             assert False, "positions must be either a torch.Tensor or a np.ndarray"

#         self.N_train = int(positions.shape[0]*frac_train)

#         self.train_times = self.times #[:self.N_train]
#         self.test_times = self.times #[self.N_train:]
#         self.train_positions = self.true_positions[:self.N_train]
#         self.test_positions = self.true_positions[self.N_train:]
#         self.n_samples = n_samples
#         self.batch_time = batch_time
#         self.dt = dt
#         self.total_length = total_length

#     def get_batch(self):
#         index = np.random.randint(0, self.N_train, self.n_samples)
#         s = torch.from_numpy(np.random.choice(np.arange(self.train_times.shape[0] - self.batch_time, dtype=np.int64), self.n_samples, replace=False))
#         batch_y0 = self.train_positions[index, s]  # (M, D)
#         batch_t = self.train_times[:self.batch_time]  # (T)
#         batch_y = torch.stack([self.train_positions[index, s + i] for i in range(self.batch_time)], dim=0)  # (T, M, D)
#         return batch_y0, batch_t, batch_y