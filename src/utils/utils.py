import torch
import numpy as np




def gaussian_density(x, mu, sigma):
    return torch.exp(-torch.norm(torch.tensor(x - mu).float(), dim=1)**2/(2*sigma**2))/(2*np.pi*sigma**2)


def add_spatial_encoding(gaussian_dataset):
    n_images = len(gaussian_dataset)

    
    if len(gaussian_dataset[0]) == 2:
        size = gaussian_dataset[0][0].shape[1]

    elif len(gaussian_dataset[0]) == 1:
        size = gaussian_dataset[0][0].shape[0]

    else:
        raise ValueError("Invalid dataset format")
    # print(size)

    # create the spatial encoding
    # create the x encoding
    x_encoding = np.linspace(np.zeros(size), np.ones(size), size, dtype=np.float64, axis=1)
    y_encoding = np.linspace(np.zeros(size), np.ones(size), size, dtype=np.float64, axis=0)

    samples = []

    for i in range(n_images):
        new_image = np.stack([gaussian_dataset[i][0].squeeze(), x_encoding, y_encoding], axis=0)
        if len(gaussian_dataset[0]) == 2:
            samples.append([new_image, gaussian_dataset[i][1]])

        elif len(gaussian_dataset[0]) == 1:
            samples.append(new_image)

            
    if len(gaussian_dataset[0]) == 2:
        return samples

    elif len(gaussian_dataset[0]) == 1:
        return np.array(samples)

def stack_dataset(gaussian_dataset):
    n_images = len(gaussian_dataset)
    size = gaussian_dataset[0][0].shape[1]

    samples = []

    for i in range(n_images):
        old_image = np.array(gaussian_dataset[i][0].squeeze())
        new_image = np.stack([old_image.copy(), old_image.copy(), old_image.copy()], axis=0)
        samples.append([new_image, gaussian_dataset[i][1]])

    return samples
