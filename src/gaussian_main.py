from pickle import MARK
from aiohttp import payload_type
import pygame
from time import time
import numpy as np 

from box import BouncingBall, GravityHoleBall

from utils import gaussian_density, add_spatial_encoding

INITIAL_SIZE = (28, 28)
WIDTH, HEIGHT = 1000, 1000
BOX_SIZE = (WIDTH, HEIGHT)
BACKGROUND_COLOR = (0, 0, 0)
RANDOM_INIT = True
MARGIN = 1
RADIUS = int(3.*WIDTH/INITIAL_SIZE[0])
RECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
INDICES_MATRIX = np.array([[i,j] for i in range(WIDTH) for j in range(HEIGHT)])


if RANDOM_INIT:
    x = np.random.randint(MARGIN, WIDTH - MARGIN)
    y = np.random.randint(MARGIN, HEIGHT - MARGIN)
    vx = 0.
    vy = 0.
    box = GravityHoleBall(x, y, vx, vy, BOX_SIZE, RADIUS)


else:
    box = GravityHoleBall(WIDTH // 2, HEIGHT // 2, 1000, -1000, BOX_SIZE, RADIUS)

background = pygame.Surface(RECT.size)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

x, y = box.move(0.)
# place the object at its initial position in the screen

t=time()

while True:

    # set the black background
    screen.fill(BACKGROUND_COLOR, RECT)

    # if I click cmd + w it closes the window
    if pygame.event.get(pygame.QUIT):
        break
    
    delta_t = time() - t
    t = time()
    # print(delta_t)
    x, y = box.move(delta_t)
    # print(box.x, box.y)
    # extract the gaussian ball image
    ball_img = gaussian_density(INDICES_MATRIX, np.array([x, y]), RADIUS).reshape(WIDTH, HEIGHT).numpy()
    ball_img = np.stack([ball_img, ball_img, ball_img], axis=-1)
    # normalize it
    ball_img = (ball_img - ball_img.min())/(ball_img.max() - ball_img.min())

    ball_img = (255*ball_img).astype(np.uint8)
    # print(ball_img.shape)
    surf = pygame.surfarray.make_surface(ball_img)
    

    # add the ball_img to the display
    screen.blit(surf, (0, 0))

    pygame.display.update()

    # img = pygame.display.get_surface()
    # print(img)
    # pygame.image.save(img, "frame.png")
    # break