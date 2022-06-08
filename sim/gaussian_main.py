import pygame
from time import time
import numpy as np 
import sys

sys.path.append("..")

from src.data.box import BouncingBall, GravityHoleBall
from src.utils.utils import gaussian_density

INITIAL_SIZE = (28, 28)
WIDTH, HEIGHT = 500, 500
BOX_SIZE = (WIDTH, HEIGHT)
BACKGROUND_COLOR = (0, 0, 0)
RANDOM_INIT = True
MARGIN_MAX = 30
MARGIN_MIN = 3
MIN_INIT_VELOCITY = 200.
RADIUS = int(3.*WIDTH/INITIAL_SIZE[0])
RECT = pygame.Rect(0, 0, WIDTH, HEIGHT)
INDICES_MATRIX = np.array([[i,j] for i in range(WIDTH) for j in range(HEIGHT)])

box = GravityHoleBall(WIDTH // 2, HEIGHT // 2, 1000, -1000, BOX_SIZE, RADIUS)

if RANDOM_INIT:

    # x = np.random.uniform(0, WIDTH - 1 - 0)
    # y = np.random.uniform(0, HEIGHT - 1 - 0)
    # # # x**2 + y**2 < (width-margin)**2
    # while (x - box.gravity_position[0])**2 + (y - box.gravity_position[1])**2 > (MARGIN_MAX)**2 or \
    #     (x - box.gravity_position[0])**2 + (y - box.gravity_position[1])**2 < (MARGIN_MIN)**2:

    #     print("No")

    #     x = np.random.uniform(0, WIDTH - 1)
    #     y = np.random.uniform(0, HEIGHT - 1)

    x, y = np.random.uniform(MARGIN_MIN, WIDTH - 1 - MARGIN_MIN, 2)

    vx, vy = np.random.uniform(-MIN_INIT_VELOCITY, MIN_INIT_VELOCITY, 2)
    # vx = 0.
    # vy = 0.
    box.reset(x0=x, y0=y, vx0=vx, vy0=vy)



background = pygame.Surface(RECT.size)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

x, y = box.move(0.)
# place the object at its initial position in the screen
print(x, y)
print(box.x, box.y)
print(box.vx, box.vy)
print(box.velocity)
t=time()

init_velo = np.abs(box.velocity.copy())

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

    print(init_velo)
    init_velo = np.maximum(init_velo, np.abs(box.velocity))
    
    

    # img = pygame.display.get_surface()
    # print(img)
    # pygame.image.save(img, "frame.png")
    # break