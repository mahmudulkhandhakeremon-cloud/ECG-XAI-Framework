import pygame
import random

pygame.init()

WIDTH, HEIGHT = 600, 400
CELL = 20

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("রসে ভরা নাগিন গেম---নির্মাতা নামে প্রকৌশলী কামে মিস্তিরি মাহমুদুল খন্দকার ইমন")

clock = pygame.time.Clock()

WHITE = (255,255,255)
GREEN = (0,255,0)
RED = (255,0,0)
BLACK = (0,0,0)

snake = [(100,100)]
dx = CELL
dy = 0

food = (
    random.randrange(0, WIDTH, CELL),
    random.randrange(0, HEIGHT, CELL)
)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and dy == 0:
                dx, dy = 0, -CELL
            elif event.key == pygame.K_DOWN and dy == 0:
                dx, dy = 0, CELL
            elif event.key == pygame.K_LEFT and dx == 0:
                dx, dy = -CELL, 0
            elif event.key == pygame.K_RIGHT and dx == 0:
                dx, dy = CELL, 0

    head = (snake[0][0] + dx, snake[0][1] + dy)

    # Game Over
    if (
        head[0] < 0 or head[0] >= WIDTH or
        head[1] < 0 or head[1] >= HEIGHT or
        head in snake
    ):
        break

    snake.insert(0, head)

    if head == food:
        food = (
            random.randrange(0, WIDTH, CELL),
            random.randrange(0, HEIGHT, CELL)
        )
    else:
        snake.pop()

    screen.fill(BLACK)

    pygame.draw.rect(screen, RED, (*food, CELL, CELL))

    for part in snake:
        pygame.draw.rect(screen, GREEN, (*part, CELL, CELL))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
