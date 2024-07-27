import os
import random
import numpy as np
import pygame
import neat
import pickle

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH = 400
HEIGHT = 600
BIRD_RADIUS = 20
PIPE_WIDTH = 50
PIPE_GAP = 200
PIPE_SPEED = 5
PIPE_SPAWN_INTERVAL = 90  # Frames between pipe spawns

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Initialize the game window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird AI")

class Bird:
    def __init__(self, y):
        self.x = 50
        self.y = y
        self.velocity = 0
        self.gravity = 0.8
        self.jump_strength = -10

    def jump(self):
        self.velocity = self.jump_strength

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def draw(self):
        pygame.draw.circle(window, BLACK, (int(self.x), int(self.y)), BIRD_RADIUS)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(150, HEIGHT - 150 - PIPE_GAP)

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self):
        pygame.draw.rect(window, GREEN, (self.x, 0, PIPE_WIDTH, self.height))
        pygame.draw.rect(window, GREEN, (self.x, self.height + PIPE_GAP, PIPE_WIDTH, HEIGHT - self.height - PIPE_GAP))

def eval_genomes(genomes, config):
    nets = []
    birds = []
    ge = []
    pipes = [Pipe(WIDTH)]
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(HEIGHT // 2))
        ge.append(genome)
        genome.fitness = 0

    clock = pygame.time.Clock()
    score = 0
    frame_count = 0

    run = True
    while run and len(birds) > 0:
        clock.tick(60)
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                pipe_ind = 1

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), 
                                       abs(bird.y - (pipes[pipe_ind].height + PIPE_GAP)), 
                                       pipes[pipe_ind].x - bird.x))

            if output[0] > 0.5:
                bird.jump()

        # Move and remove pipes
        for pipe in pipes:
            pipe.move()

        if pipes and pipes[0].x + PIPE_WIDTH < 0:
            pipes.pop(0)
            score += 1
            for g in ge:
                g.fitness += 5

        # Add new pipe
        if frame_count % PIPE_SPAWN_INTERVAL == 0:
            pipes.append(Pipe(WIDTH))

        # Check for collisions
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if (pipe.x < bird.x + BIRD_RADIUS < pipe.x + PIPE_WIDTH and
                    (bird.y - BIRD_RADIUS < pipe.height or bird.y + BIRD_RADIUS > pipe.height + PIPE_GAP)):
                    ge[x].fitness -= 1
                    nets.pop(x)
                    ge.pop(x)
                    birds.pop(x)

        for x, bird in enumerate(birds):
            if bird.y + BIRD_RADIUS > HEIGHT or bird.y - BIRD_RADIUS < 0:
                ge[x].fitness -= 1
                nets.pop(x)
                ge.pop(x)
                birds.pop(x)

        window.fill(WHITE)
        for pipe in pipes:
            pipe.draw()
        for bird in birds:
            bird.draw()

        # Display the current score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {score}', True, BLACK)
        window.blit(score_text, (10, 10))

        pygame.display.update()

        if score > 100:
            run = False

    return score

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    best_score = 0
    generation = 0
    
    while best_score < 100:
        generation += 1
        winner = p.run(eval_genomes, 1)
        best_score = max(genome.fitness for genome in p.population.values())
        
        print(f"Generation {generation}: Best score = {best_score}")
        
        # Save the best genome
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        if best_score >= 100:
            print(f"Target score reached in generation {generation}!")
            break

    print('\nBest genome:\n{!s}'.format(winner))

    # Replay the best genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    bird = Bird(HEIGHT // 2)
    pipes = [Pipe(WIDTH)]
    score = 0
    clock = pygame.time.Clock()
    frame_count = 0

    while True:
        clock.tick(60)
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        output = winner_net.activate((bird.y, abs(bird.y - pipes[0].height), 
                                      abs(bird.y - (pipes[0].height + PIPE_GAP)), 
                                      pipes[0].x - bird.x))
        if output[0] > 0.5:
            bird.jump()

        bird.move()
        for pipe in pipes:
            pipe.move()

        if pipes and pipes[0].x + PIPE_WIDTH < 0:
            pipes.pop(0)
            score += 1

        if frame_count % PIPE_SPAWN_INTERVAL == 0:
            pipes.append(Pipe(WIDTH))

        if (bird.y + BIRD_RADIUS > HEIGHT or bird.y - BIRD_RADIUS < 0 or
            any(pipe.x < bird.x + BIRD_RADIUS < pipe.x + PIPE_WIDTH and
                (bird.y - BIRD_RADIUS < pipe.height or bird.y + BIRD_RADIUS > pipe.height + PIPE_GAP)
                for pipe in pipes)):
            break

        window.fill(WHITE)
        for pipe in pipes:
            pipe.draw()
        bird.draw()

        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {score}', True, BLACK)
        window.blit(score_text, (10, 10))

        pygame.display.update()

    print(f"Final Score: {score}")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)