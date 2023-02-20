import numpy as np

class Object:
    """
    A simple object with a position and a velocity

    Parameters
    ----------
    x : float, initial x position
    y : float, initial y position
    vx : float, initial x velocity
    vy : float, initial y velocity
    size : int or tuple, size of the object
    """
    def __init__(self, x, y, vx, vy, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        assert isinstance(size, int) or len(size) == 2

        if isinstance(size, int):
            self.size = np.array([size, size])
        else:
            self.size = size

    # def move(self, dt):
    #     self.x += self.vx * dt
    #     self.y += self.vy * dt
    #     return self.x, self.y



class Box:
    """
    Class Box that contains an object at a given position and velocity

    Parameters
    ----------
    x : float, initial x position
    y : float, initial y position
    vx : float, initial x velocity
    vy : float, initial y velocity
    box_size : int or tuple, size of the box
    object_size : int or tuple, size of the object
    """
    def __init__(self, x, y, vx, vy, box_size, object_size):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        vx = np.array(vx, dtype=np.float64)
        vy = np.array(vy, dtype=np.float64)

        
        self.x0 = x
        self.y0 = y

        self.x = np.copy(x)
        self.y = np.copy(y)

        self.vx0 = vx
        self.vy0 = vy
        self.vx = np.copy(vx)
        self.vy = np.copy(vy)

        assert isinstance(box_size, int) or len(box_size) == 2

        if isinstance(box_size, int):
            self.box_size = np.array([box_size, box_size])
        else:
            self.box_size = np.array(box_size)

        self.object = Object(x, y, vx, vy, object_size)

        assert not self.out_of_box(self.x, self.y)

    def above(self, x, y):
        return y + self.object.size[1] > self.box_size[1]

    def below(self, x, y):
        return y < 0

    def left(self, x, y):
        return x < 0
    
    def right(self, x, y):
        return x + self.object.size[0] > self.box_size[0]
    
    def out_of_box(self, x, y):
        return x < 0 or x + self.object.size[0] > self.box_size[0] or y < 0 or y + self.object.size[1] > self.box_size[1]

    def reset(self, **kwargs):

        if 'x0' in kwargs:
            self.x0 = kwargs['x0']
        if 'y0' in kwargs:
            self.y0 = kwargs['y0']
        if 'vx0' in kwargs:
            self.vx0 = kwargs['vx0']
        if 'vy0' in kwargs:
            self.vy0 = kwargs['vy0']

        self.x = np.copy(self.x0)
        self.y = np.copy(self.y0)
        self.vx = np.copy(self.vx0)
        self.vy = np.copy(self.vy0)

        return self.x, self.y

class BouncingBall(Box):
    """
    Class BouncingBall that contains an object at a given position and velocity 
    (randomly generated on the image) and that bounces on the walls of the box.

    Parameters
    ----------
    x : float, initial x position
    y : float, initial y position
    vx : float, initial x velocity
    vy : float, initial y velocity
    box_size : int or tuple, size of the box
    object_size : int or tuple, size of the object
    """
    def __init__(self, x, y, vx, vy, box_size, object_size):
        super().__init__(x, y, vx, vy, box_size, object_size)

    def move(self, dt):
        # compute the predicted mouvement
        x = self.x + self.vx * dt
        y = self.y + self.vy * dt
        
        # if the object is inside it won't change
        # the signs of vx and vy
        signs = np.array([1., 1.])

        # Check if the object is out of the box and change
        # the sign of the velocity depending on the side
    
        if self.above(x, y):
            signs *= np.array([1., -1.])
            y = self.box_size[1] - self.object.size[1]
            dt -= (y - self.y) / self.vy
            self.y = y

        if self.below(x, y):
            signs *= np.array([1., -1.])
            y = 0
            dt -= (y - self.y) / self.vy
            self.y = y
        
        if self.left(x, y):
            signs *= np.array([-1., 1.])
            x = 0
            dt -= (x - self.x) / self.vx
            self.x = x

        if self.right(x, y):
            signs *= np.array([-1., 1.])
            x = self.box_size[0] - self.object.size[0]
            dt -= (x - self.x) / self.vx
            self.x = x
        # print(self.x)
        # print(self.vx * dt * signs[0])
        self.vx *= signs[0]
        self.vy *= signs[1]

        self.x += self.vx * dt 
        self.y += self.vy * dt
    

        return self.x, self.y

# GRAVITY_ACCELERATION = 3.0e1
# GRAVITY_ACCELERATION = 9.0e1
GRAVITY_ACCELERATION = 1.0e2


# FRICTION = 1.
FRICTION = 1.5

class GravityHoleBall(Box):
    """
    Class GravityHoleBall that contains an object at a given position and velocity 
    (randomly generated on the image) and that is attracted by a gravity hole.

    Parameters
    ----------
    x : float, initial x position
    y : float, initial y position
    vx : float, initial x velocity
    vy : float, initial y velocity
    box_size : int or tuple, size of the box
    object_size : int or tuple, size of the object
    gravity_position : tuple, position of the gravity hole, default is the center of the box
    """
    def __init__(self, x, y, vx, vy, box_size, object_size, gravity_position=None):
        super().__init__(x, y, vx, vy, box_size, object_size)

        if gravity_position is None:
            if isinstance(self.box_size, int):
                self.gravity_position = np.array([self.box_size / 2., self.box_size / 2. ])

            elif isinstance(self.box_size, list) or isinstance(self.box_size, tuple) or isinstance(self.box_size, np.ndarray):
                self.gravity_position = np.array(self.box_size) / 2.

        else:
            self.gravity_position = np.array(gravity_position)

        self.position = np.array([self.x, self.y])
        self.velocity = np.array([self.vx, self.vy])
        
    def move(self, dt):

        gravity_vector = self.gravity_position - self.position

        self.velocity += GRAVITY_ACCELERATION * dt * gravity_vector - FRICTION * self.velocity * dt

        self.vx = self.velocity[0]
        self.vy = self.velocity[1]

        self.x += self.vx * dt
        self.y += self.vy * dt

        self.position = np.array([self.x, self.y])

        return self.x, self.y

    def reset(self, **kwargs):
        if 'x0' in kwargs:
            self.x0 = kwargs['x0']
        if 'y0' in kwargs:
            self.y0 = kwargs['y0']
        if 'vx0' in kwargs:
            self.vx0 = kwargs['vx0']
        if 'vy0' in kwargs:
            self.vy0 = kwargs['vy0']

        self.x = np.copy(self.x0)
        self.y = np.copy(self.y0)
        self.vx = np.copy(self.vx0)
        self.vy = np.copy(self.vy0)

        self.position = np.copy(np.array([self.x, self.y]))
        self.velocity = np.copy(np.array([self.vx, self.vy]))

        return self.x, self.y
        

    