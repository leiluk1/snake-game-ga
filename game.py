import pygame
import sys
import random
import numpy as np
import time
import copy
from numpy.core.multiarray import ndarray
from pygame.math import Vector2
from typing import List


def relu(a):
    return a * (a > 0)


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x), axis=1)


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
    def __init__(self, thetas1: ndarray, thetas2: ndarray):
        self.snake = Snake()
        self.food = Food()
        self.allowed_steps = 200

        self.thetas1 = thetas1
        self.thetas2 = thetas2

    def look_in_direction(self, direction: Vector2):
        food_counter, obstacle_counter = -1, -1

        head_copy = self.snake.body[0].copy()
        counter = 0
        while not (head_copy.x < 0 or head_copy.x >= BLOCK_SIZE or
                   head_copy.y < 0 or head_copy.y >= BLOCK_SIZE):
            counter += 1
            head_copy += direction
            for part in self.snake.body[1:]:
                if part == head_copy:
                    obstacle_counter = counter
                    return obstacle_counter # TODO Добавлено дополнительно
            # if head_copy == self.food.pos:
            #     food_counter = counter
        if obstacle_counter == -1:
            obstacle_counter = counter
        return obstacle_counter
        #return food_counter, obstacle_counter

    def get_distances(self):  # returns distances to food and closest obstacle for 3 directions
        dir = self.snake.dir
        head = self.snake.body[0]
        a,b = self.food.pos.x - head.x, self.food.pos.y - head.y
        c = self.look_in_direction(Vector2(-dir.y, dir.x))
        d = self.look_in_direction(dir)
        e = self.look_in_direction(Vector2(dir.y, -dir.x))
        #c, d = self.look_in_direction(dir)
        #e, f = self.look_in_direction(Vector2(dir.y, -dir.x))
        return np.array([a, b, c, d, e])
        #return np.array([a, b, c, d, e, f])

    def decision(self):
        input = self.get_distances()
        input = np.append(input, 1)
        layer1 = input @ self.thetas1
        layer1 = relu(layer1)
        layer1 = np.append(layer1, 1)
        output = layer1 @ self.thetas2
        output = output.reshape([1, -1])
        output = softmax(output)
        return np.argmax(output) - 1

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

    def update(self) -> bool:

        self.snake.move_snake()
        self.check_objects()
        if self.allowed_steps <= 0:
            print("end of allowed steps")
            return True
        if self.check_out_of_screen():
            return True
        if self.check_body_collision():
            return True
        return False

    def check_objects(self):
        snake_head = self.snake.body[0]
        if snake_head == self.food.pos:
            self.allowed_steps = 200
            self.food.random_food_pos()
            self.snake.grow()
        else:
            self.allowed_steps -= 1

    def check_out_of_screen(self):
        snake_head = self.snake.body[0]
        if (snake_head.x < 0 or snake_head.x >= BLOCK_SIZE or
                snake_head.y < 0 or snake_head.y >= BLOCK_SIZE):
            pygame.quit()
            print("Out of screen")
            return True
        return False

    def check_body_collision(self):
        snake_head = self.snake.body[0]
        for part in self.snake.body[1:]:
            if part == snake_head:
                pygame.quit()
                print("Body collision")
                return True
        return False


def play_game(genes, with_display=False):
    global screen, font
    screen = pygame.display.set_mode((BLOCK_SIZE ** 2, BLOCK_SIZE ** 2))
    pygame.font.init()
    font = pygame.font.SysFont('umeminchos3', 20)
    pygame.display.set_caption('Snake game')

    game = Game(genes[0], genes[1])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        decision = game.decision()
        if decision == -1:
            dir = game.snake.dir
            game.snake.dir = Vector2(-dir.y, dir.x)
        elif decision == 1:
            dir = game.snake.dir
            game.snake.dir = Vector2(dir.y, -dir.x)

        if game.update():
            return len(game.snake.body) - 2
        if with_display:
            game.grid()
            game.place_objects()
            game.place_score()

            pygame.display.update()
        time.sleep(0.01)


BLOCK_SIZE = 20
GREEN = (120, 177, 90)
LIGHT_GREEN = (133, 187, 101)
BLACK = (20, 67, 76)
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((BLOCK_SIZE ** 2, BLOCK_SIZE ** 2))
pear = pygame.image.load('objects/pear_fruit.png').convert_alpha()

pygame.font.init()
font = pygame.font.SysFont('umeminchos3', 20)

pygame.display.set_caption('Snake game')

# ===========
best_fitness = 0
population_size = 10  # how many snakes
mutation_probability = 0.5
generations = 100
crossover_rate = 0.5  # probability that some weight of NN in parent 1 will be replaced by that of parent 2 in children


# =============

def generate_random_genoms():
    return [np.random.randn(6, 4), np.random.randn(5, 3)]
    #return [np.random.randn(7, 4), np.random.randn(5, 3)]


def generate_population():  # Returns list of ndarrays
    return [generate_random_genoms() for _ in range(population_size)]


def calculate_population_fitnesses(population: List[List[ndarray]]):
    results = []
    for ind_genes in population:
        results.append(play_game(ind_genes, True))  # play a game, record the result
    return results


def selection(population: List[List[ndarray]], fitness: List, n: int) -> List[List[ndarray]]:
    # Roulette wheel selection
    fitness_sum = sum(fitness)
    if fitness_sum > 3:
        probs = [fitness[i] / fitness_sum for i in range(len(fitness))]

        indices = np.argsort(probs)
        population = [population[i] for i in indices]
        probs = np.sort(probs)
        probs = np.cumsum(probs)
        selected_inds = []
        for i in range(n):
            p = np.random.uniform(0, 1)
            index = 0
            while probs[index] < p:
                index += 1
            selected_inds.append(population[index])
        return selected_inds
    else:
        return random.sample(population, n)

def crossover_ind(parents: List[List[ndarray]]) -> List[ndarray]:
    parent1 = parents[np.random.randint(0, len(parents))]
    parent2 = parents[np.random.randint(0, len(parents))]
    children = generate_random_genoms()
    for index_of_table in range(len(children)):
        for row in range(children[index_of_table].shape[0]):
            for column in range(children[index_of_table].shape[1]):
                p = np.random.uniform(0, 1)
                if p < crossover_rate:
                    children[index_of_table][row][column] = parent2[index_of_table][row][column]
                else:
                    children[index_of_table][row][column] = parent1[index_of_table][row][column]
    return children


def crossover_pop(parents: List[List[ndarray]], n: int) -> List[List[ndarray]]:
    offsprings = []
    for i in range(n):
        offsprings.append(crossover_ind(parents))
    return offsprings


def mutate(offsprings: List[List[ndarray]], p: float) -> List[List[ndarray]]:
    mutated_offsprings = []
    for offspring in offsprings:  # all offsprings
        cur_offspring = copy.deepcopy(offspring)
        for weight in cur_offspring:  # all lists
            for _ in range(int(weight.shape[0] * weight.shape[1] * p)):
                row = random.randint(0, weight.shape[0] - 1)
                col = random.randint(0, weight.shape[1] - 1)
                weight[row, col] += random.uniform(-0.1, 0.1)
        mutated_offsprings.append(cur_offspring)
    return mutated_offsprings

population = generate_population()
# counter = 0
# best_answer = []
for gen_num in range(generations):
    fitness = calculate_population_fitnesses(population)
    print(fitness)
    best_fitness = np.max(fitness)
    print("Generation {}, best fitness: {}".format(gen_num, np.max(fitness)))
    parents = selection(population, fitness, int(population_size / 2))
    offsprings = crossover_pop(parents, int(population_size / 2))
    offsprings = mutate(offsprings, mutation_probability)
    population = offsprings + parents  # new population is the combination of offsprings and parents

fitness = calculate_population_fitnesses(population)
print("Last generation, best fitness: {}".format(np.max(fitness)))
print("best weights: {}".format(population[np.argmax(fitness)]))
answer = population[np.argmax(fitness)]
