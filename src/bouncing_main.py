import pygame
from time import time
import numpy as np 

from box import BouncingBall, GravityHoleBall

WIDTH, HEIGHT = 800, 800
BOX_SIZE = (WIDTH, HEIGHT)
BACKGROUND_COLOR = (0, 0, 0)
RANDOM_INIT = True

RECT = pygame.Rect(0, 0, WIDTH, HEIGHT)

background_img = pygame.image.load("background.png")
# resize it to fit the screen dimensions
background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))

box_img = pygame.image.load('shark.png')
box_img = pygame.transform.scale(box_img, (80, 64))
OBJ_SIZE = box_img.get_size()

if RANDOM_INIT:
    x = np.random.randint(0, WIDTH - OBJ_SIZE[0])
    y = np.random.randint(0, HEIGHT - OBJ_SIZE[1])
    vx = 0. #np.random.uniform(0, 1)*WIDTH*np.random.choice([-1, 1])*0.5
    vy = 0. #np.random.uniform(0, 1)*HEIGHT*np.random.choice([-1, 1])*0.5
    box = GravityHoleBall(x, y, vx, vy, BOX_SIZE, OBJ_SIZE)


else:
    box = GravityHoleBall(WIDTH // 2, HEIGHT // 2, 1000, -1000, BOX_SIZE, OBJ_SIZE)

background = pygame.Surface(RECT.size)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

x, y = box.move(0.)
# place the object at its initial position in the screen
screen.blit(box_img, (x, y))


t=time()

while True:

    # set the black background
    screen.fill(BACKGROUND_COLOR, RECT)
    # draw the background image
    screen.blit(background_img, (0, 0))

    # if I click cmd + w it closes the window
    if pygame.event.get(pygame.QUIT):
        break
    
    delta_t = time() - t
    t = time()
    print(delta_t)
    x, y = box.move(delta_t)
    screen.blit(box_img, (x, y))
    # print(box.x, box.y)

    pygame.display.update()

    # img = pygame.display.get_surface()
    # print(img)
    # pygame.image.save(img, "frame.png")
    # break