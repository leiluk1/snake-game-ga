## Set up
To run this project, run the following commands in the repo root directory:
1. Create virtual environment
    ```
    python3 -m venv venv
    source ./venv/bin/activate
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Make sure you have a compatible version of Python 3.9.13 before running the code since it was tested on that version.


# Snake Game Using Genetic Algorithm

The project objective is to implement a version of classical Snake game with PyGame and apply genetic algorithm to teach neural network play the game. neural network consists of one hidden layer of size 15. Weights for this NN are determined by GA. The input to neural network is information from 7 directions(all are got by rotating by 45 degrees and excluding 180 degrees) about distance to food(0 if no food in the direction), distance to body(0 if no body in the direction) and distance to the wall. Therefore, NN takes 21 value as input plus bias term. 

GA includes selection, crossover and mutation operations. Selection operationg selects the best n/2 individuals by fitness, where n is size of the population. Crossover takes 2 random parents and generated an offspring with every gene taken from either parent 1 with probability p or parent 2 with probability 1-p. Mutation changes genes by some uniformly distributed value from -0.1 to 0.1 with specified probability. The initial population consist of individuals with genes sampled from normal distribution with mean 0 and variance 0.6.


This project is intended for the "Nature Inspired Computing" course at Innopolis University.

## Game demo


https://user-images.githubusercontent.com/88292173/235656591-8ae5ab53-94de-4c7e-b864-01523743054e.MP4

## Sources
- Snake game visual inspired by Clear Code [tutorial](https://www.youtube.com/watch?app=desktop&v=QFvqStqPCRU&pp=ygUMUHlnYW1lIHNuYWtl)
