import cv2
import numpy as np

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tqdm.notebook import trange

from src.utils.utils import gaussian_density


class Ball:
    """
    Class for a ball in 2D space that follows a given move function

    Parameters
    ----------
    x0 : np.array of shape (2,), initial position of the ball
    r : int, radius of the ball
    move_func : function, function that takes as input the time and the position of the ball and returns the velocity of the ball
    """
    def __init__(self, x0, r, move_func):
        self.x0 = x0
        self.r = int(np.array(r))
        self.x = np.copy(x0)
        self.t = 0
        self.move_func = move_func

    def reset(self):
        self.x = np.copy(self.x0)
        self.t = 0

    def step(self, dt):
        self.t += dt
        x = np.copy(self.x)
        self.x += dt*self.move_func(self.t, x)
        
        return self.x



class Image(Ball):
    """
    Class wrapper of the Ball class that returns an image of the ball instead of the position

    Parameters
    ----------
    x0 : np.array of shape (2,), initial position of the ball
    r : int, radius of the ball
    move_func : function, function that takes as input the time and the position of the ball and returns the velocity of the ball
    size : int, size of the image
    border : tuple of two floats or int, border of the image
    gaussian : bool, if True the image is a gaussian density, if False the image is a circle
    """
    def __init__(self, x0, r, move_func, size, border=[2,2], gaussian=True):
        super(Image, self).__init__(x0, r, move_func)
        self.size = size
        self.border = border
        if isinstance(self.border, int): 
            self.border = (self.border, self.border)
        self.image_dim = (size, size, 3)
        self.image = np.zeros(self.image_dim, dtype=np.uint8)
        self.r = r
        self.gaussian = gaussian

        x = self.compute_position_on_image(self.x)
        self.indices_matrix = np.array([[i,j] for i in range(size) for j in range(size)])
    
        if gaussian:
            image = np.zeros((size, size), dtype=np.uint8)
            # compute the density over the image and normalize it
            image = gaussian_density(self.indices_matrix, x, r).numpy().copy()
            image = (image - image.min()) / (image.max() - image.min())
            self.image = image.reshape(1, size, size)


        else:
            self.image = cv2.circle(np.zeros(self.image_dim, dtype=np.uint8), (x[0], x[1]), self.r, (255, 255, 255), -1)


    def compute_position_on_image(self, x):
        x = (x - self.border[0])/(self.border[1] - self.border[0])
        x = x*(self.size-1)
        x = x.astype(np.uint8)
        x = np.clip(x, 0, self.size-1)
        return x

    def compute_position_on_image_float(self, x):
        x = (x - self.border[0])/(self.border[1] - self.border[0])
        x = x*(self.size-1)
        x = x.astype(np.float32)
        x = np.clip(x, 0, self.size-1)
        return x

    def update_image(self):
        x = self.compute_position_on_image_float(self.x)
        if self.gaussian:
            image = np.zeros((self.size, self.size), dtype=np.uint8)
            # x = np.random.randint(self.size, size=(2))
            # compute the density over the image and normalize it
            image = gaussian_density(self.indices_matrix, x, self.r).numpy().copy()
            image = (image - image.min()) / (image.max() - image.min())
            self.image = image.reshape(self.size, self.size, 1)

        else:
            self.image = cv2.circle(np.zeros(self.image_dim, dtype=np.uint8), (x[0], x[1]), self.r, (255, 255, 255), -1)

    def display(self):
        self.update_image()
        plt.imshow(self.image)
        plt.show()

    def forward(self, dt):
        x = self.step(dt)
        self.update_image()
        return x

    def animate(self, dt, frames=100, interval=20, repeat_delay=0):
        
        self.reset()
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.size), ylim=(0, self.size))
        ax.imshow(self.image)

        def init():
            self.reset()
            ax.imshow(self.image)
            return ax,

        def update(i):
            self.forward(dt)
            ax.imshow(self.image)
            return ax,
            

        anim = FuncAnimation(fig, update, init_func=init, frames=trange(0,frames), interval=interval, repeat_delay=repeat_delay)
        # plt.show()
        anim.save('images/AE/ball_animation.gif', writer='pillow')
        plt.close()

    def generate_samples(self, dt, frames=100, conv=False):
        self.reset()


        if frames > 0:
            if conv:
                samples = np.zeros((frames, 1, self.image_dim[0], self.image_dim[1]))

            else:
                samples = np.zeros((frames, self.image_dim[0], self.image_dim[1]))

            for i in range(frames):
                self.forward(dt)
                if conv:
                    samples[i, 0] = self.image[:,:,0]

                else:
                    samples[i] = self.image[:,:,0]

            return samples

        else:
            samples = []
            current_x = np.copy(self.x)
            while True:
                x = self.forward(dt)
                if np.linalg.norm(x - current_x) < 1E-2*dt:
                    break
                current_x = np.copy(x)
                if conv: 
                    samples.append(np.expand_dims(self.image[:,:,0], axis=0))
                else:
                    samples.append(self.image[:,:,0])
                
            return np.array(samples)
    
    def generate_trajectory_positions(self, dt, frames=100):
        self.reset()

        if frames > 0:
            
            samples = np.zeros((frames, 2))

            for i in range(frames):
                x = self.forward(dt)
                x = self.compute_position_on_image_float(x)
                samples[i] = x

            return samples

        else:
            samples = []
            current_x = np.copy(self.x)
            while True:
                x = self.forward(dt)
                x = self.compute_position_on_image_float(x)
                if np.linalg.norm(x - current_x) < 1E-2*dt:
                    break
                current_x = np.copy(x)
                samples.append(x)
                
            return np.array(samples)

def move_fun_circle(t, x, w=1, exp_decay=1.):
    """
    Move function for a circle with angular velocity w and exponential decay exp_decay

    Parameters
    ----------
    t : np.array, time
    x : not used
    w : float, angular velocity
    exp_decay : float, exponential decay
    """
    return np.exp(-exp_decay*t)*np.array([-w*np.sin(w*t), w*np.cos(w*t)], dtype=np.float32)    



# class Ball:
#     def __init__(self, x0, r, move_func):
#         self.x0 = x0
#         self.r = int(np.array(r))
#         self.x = np.copy(x0)
#         self.t = 0
#         self.move_func = move_func

#     def reset(self):
#         self.x = np.copy(self.x0)
#         self.t = 0

#     def step(self, dt):
#         self.t += dt
#         x = np.copy(self.x)
#         self.x += dt*self.move_func(self.t, x)
        
#         return self.x

    

# class Image(Ball):
#     def __init__(self, x0, r, move_func, size, border=2):
#         super(Image, self).__init__(x0, r, move_func)
#         self.size = size
#         self.border = border
#         self.image_dim = (size, size, 3)
#         self.image = np.zeros(self.image_dim, dtype=np.uint8)
        
#         x = self.compute_position_on_image(self.x)
#         self.image = cv2.circle(np.zeros(self.image_dim, dtype=np.uint8), (x[0], x[1]), self.r, (255, 255, 255), -1)


#     def compute_position_on_image(self, x):
#         x = (x - self.border[0])/(self.border[1] - self.border[0])
#         x = x*(self.size-1)
#         x = x.astype(np.uint8)
#         x = np.clip(x, 0, self.size-1)

#         return x

#     def update_image(self):
#         x = self.compute_position_on_image(self.x)
#         self.image = cv2.circle(np.zeros(self.image_dim, dtype=np.uint8), (x[0], x[1]), self.r, (255, 255, 255), -1)


#     def display(self):
#         self.update_image()
#         plt.imshow(self.image)
#         plt.show()

#     def forward(self, dt):
#         x = self.step(dt)
#         self.update_image()
#         return x

#     def animate(self, dt, frames=100, interval=20, repeat_delay=0):
        
#         self.reset()
#         fig = plt.figure()
#         ax = plt.axes(xlim=(0, self.size), ylim=(0, self.size))
#         ax.imshow(self.image)

#         def init():
#             self.reset()
#             ax.imshow(self.image)
#             return ax,

#         def update(i):
#             self.forward(dt)
#             ax.imshow(self.image)
#             return ax,
            

#         anim = FuncAnimation(fig, update, init_func=init, frames=trange(0,frames), interval=interval, repeat_delay=repeat_delay)
#         # plt.show()
#         anim.save('images/ODE/ball_animation.gif', writer='pillow')
#         plt.close()

#     def generate_samples(self, dt, frames=100, conv=False):
#         self.reset()


#         if frames > 0:
#             if conv:
#                 samples = np.zeros((frames, 1, self.image_dim[0], self.image_dim[1]))

#             else:
#                 samples = np.zeros((frames, self.image_dim[0], self.image_dim[1]))

#             for i in range(frames):
#                 self.forward(dt)
#                 if conv:
#                     samples[i, 0] = self.image[:,:,0]

#                 else:
#                     samples[i] = self.image[:,:,0]

#             return samples

#         else:
#             samples = []
#             current_x = np.copy(self.x)
#             while True:
#                 x = self.forward(dt)
#                 if np.linalg.norm(x - current_x) < 1E-2*dt:
#                     break
#                 current_x = np.copy(x)
#                 if conv: 
#                     samples.append(np.expand_dims(self.image[:,:,0], axis=0))
#                 else:
#                     samples.append(self.image[:,:,0])
                
#             return np.array(samples)


# def move_fun_circle(t, x, w=1, exp_decay=1.):
#     return np.exp(-exp_decay*t)*np.array([-w*np.sin(w*t), w*np.cos(w*t)], dtype=np.float32)    

