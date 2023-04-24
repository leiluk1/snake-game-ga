import pygame
import sys
import random
from pygame.math import Vector2

class Snake:
    def __init__(self):
        self.body = [Vector2(1, 0), Vector2(0, 0)]
        self.dir = Vector2(1, 0)
        self.is_grow_up = False
        
        self.hor_part = pygame.image.load('objects/hor_part.png').convert_alpha()
        self.vert_part = pygame.image.load('objects/vert_part.png').convert_alpha()
        
        self.down_right_part = pygame.image.load('objects/down_right_part.png').convert_alpha()
        self.down_left_part = pygame.image.load('objects/down_left_part.png').convert_alpha()
        self.up_right_part = pygame.image.load('objects/up_right_part.png').convert_alpha()
        self.up_left_part = pygame.image.load('objects/up_left_part.png').convert_alpha()
        
        self.head_up = pygame.image.load('objects/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('objects/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('objects/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('objects/head_left.png').convert_alpha()
        
        self.tail_up = pygame.image.load('objects/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('objects/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('objects/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('objects/tail_left.png').convert_alpha()
        
    def place_snake(self):
        self.pick_head_dir()
        self.pick_tail_dir()
        
        for idx, part in enumerate(self.body):
            x, y = part.x * BLOCK_SIZE, part.y * BLOCK_SIZE
            part_rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)   

            if idx == 0:
                image = self.head
                
            elif idx + 1 == len(self.body):
                image = self.tail
                
            else:
                prev, next = self.body[idx + 1] - part, self.body[idx - 1] - part
                if prev.y == next.y:
                    image = self.hor_part
                elif prev.x == next.x:
                    image = self.vert_part
                else:
                    if prev.x == 1 and next.y == 1 or prev.y == 1 and next.x == 1:
                        image = self.down_right_part
                    elif prev.x == -1 and next.y == -1 or prev.y == -1 and next.x == -1:
                        image = self.up_left_part
                    elif prev.x == -1 and next.y == 1 or prev.y == 1 and next.x == -1:
                        image = self.down_left_part
                    elif prev.x == 1 and next.y == -1 or prev.y == -1 and next.x == 1:
                        image = self.up_right_part

            screen.blit(image, part_rect)

    def move_snake(self):
        head_pos = self.body[0] + self.dir
        
        if self.is_grow_up:
            self.body.insert(0, head_pos)
            self.is_grow_up = False
        else:
            body_new_pos = self.body[:-1]
            body_new_pos.insert(0, head_pos)
            self.body = body_new_pos[:]
    
    def grow(self):
        self.is_grow_up = True
        
    def pick_head_dir(self):
        head_dir = self.body[1] - self.body[0]
        
        if head_dir == Vector2(0, 1):
            self.head = self.head_up
        if head_dir == Vector2(0, -1):
            self.head = self.head_down
        if head_dir == Vector2(-1, 0):
            self.head = self.head_right
        if head_dir == Vector2(1, 0):
            self.head = self.head_left
    
    def pick_tail_dir(self):
        tail_dir = self.body[-2] - self.body[-1]
        
        if tail_dir == Vector2(0, 1):
            self.tail = self.tail_up
        if tail_dir == Vector2(0, -1):
            self.tail = self.tail_down
        if tail_dir == Vector2(-1, 0):
            self.tail = self.tail_right
        if tail_dir == Vector2(1, 0):
            self.tail = self.tail_left
    
class Food:
    def __init__(self):
        self.random_food_pos()
    
    def place_food(self):
        x, y = self.pos.x * BLOCK_SIZE, self.pos.y * BLOCK_SIZE
        food_rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
        screen.blit(pear, food_rect)
    
    def random_food_pos(self):
        self.x = random.randint(0, BLOCK_SIZE - 1)
        self.y = random.randint(0, BLOCK_SIZE - 1)
        self.pos = Vector2(self.x, self.y)
        
class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
    
    def grid(self):
        for row in range(BLOCK_SIZE):
            for col in range(BLOCK_SIZE):
                if (row + col) % 2 == 0:
                    color = LIGHT_GREEN
                else:
                    color = GREEN
                    
                pygame.draw.rect(screen, color, [col * BLOCK_SIZE, 
                                                 row * BLOCK_SIZE, 
                                                 BLOCK_SIZE, 
                                                 BLOCK_SIZE]
                                 )
    
    def place_score(self):
        score = len(self.snake.body) - 2
        
        score_text = f'score: {score}'
        score_surface = font.render(score_text, True, BLACK)
        score_rect = score_surface.get_rect(topright=(BLOCK_SIZE ** 2 - 25, 
                                                      BLOCK_SIZE ** 2 - 25)
                                            )
        screen.blit(score_surface, score_rect)
           
    def place_objects(self):
        self.food.place_food()
        self.snake.place_snake()
    
    def update(self):
        self.snake.move_snake()
        self.check_objects()
        self.check_out_of_screen()
        self.check_body_collision()
    
    def check_objects(self):
        snake_head = self.snake.body[0]
        if snake_head == self.food.pos:
            self.food.random_food_pos() 
            self.snake.grow()
            
    def check_out_of_screen(self):
        snake_head = self.snake.body[0]
        if (snake_head.x < 0 or snake_head.x >= BLOCK_SIZE or 
                snake_head.y < 0 or snake_head.y >= BLOCK_SIZE):
            
            pygame.quit()
            sys.exit()
    
    def check_body_collision(self):
        snake_head = self.snake.body[0]
        for part in self.snake.body[1:]:
            if part == snake_head:
                pygame.quit()
                sys.exit()
                
    
BLOCK_SIZE = 20
GREEN = (120, 177, 90)
LIGHT_GREEN = (133, 187, 101)
BLACK = (20, 67, 76)
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((BLOCK_SIZE ** 2, BLOCK_SIZE ** 2))
clock = pygame.time.Clock()
pear = pygame.image.load('objects/pear_fruit.png').convert_alpha()

pygame.font.init()
font = pygame.font.SysFont('umeminchos3', 20)

pygame.display.set_caption('Snake game')

UPDATE = pygame.USEREVENT
pygame.time.set_timer(UPDATE, 150)

game = Game()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            
        if event.type == UPDATE:
            game.update()  
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if game.snake.dir.y != 1:
                    game.snake.dir = Vector2(0, -1)
            if event.key == pygame.K_DOWN:
                if game.snake.dir.y != -1:
                    game.snake.dir = Vector2(0, 1)
            if event.key == pygame.K_RIGHT:
                if game.snake.dir.x != -1:
                    game.snake.dir = Vector2(1, 0)
            if event.key == pygame.K_LEFT:
                if game.snake.dir.x != 1:
                    game.snake.dir = Vector2(-1, 0)
                
    game.grid()
    game.place_objects()
    game.place_score()

    pygame.display.update()
    clock.tick(60)
