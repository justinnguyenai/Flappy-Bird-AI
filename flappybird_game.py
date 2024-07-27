import pygame
import random
import neat
import pickle
import os
import sys

# Initialize Pygame
pygame.init()

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

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
BLUE = (0, 0, 255)

# Initialize the game window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

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

def load_ai():
    config_path = os.path.join(current_dir, 'flappybird_config-feedforward.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    genome_path = os.path.join(current_dir, 'flappybird_best_genome.pkl')
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net

def draw_button(button, text):
    pygame.draw.rect(window, BLUE, button)
    font = pygame.font.Font(None, 36)
    text_surf = font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=button.center)
    window.blit(text_surf, text_rect)

def game_loop(is_player):
    bird = Bird(HEIGHT // 2)
    pipes = [Pipe(WIDTH)]
    score = 0
    clock = pygame.time.Clock()
    frame_count = 0

    if not is_player:
        ai_brain = load_ai()

    running = True
    while running:
        clock.tick(60)
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and is_player:
                if event.key == pygame.K_SPACE:
                    bird.jump()

        if not is_player:
            pipe_ind = 0
            if len(pipes) > 1 and bird.x > pipes[0].x + PIPE_WIDTH:
                pipe_ind = 1

            output = ai_brain.activate((bird.y, abs(bird.y - pipes[pipe_ind].height), 
                                        abs(bird.y - (pipes[pipe_ind].height + PIPE_GAP)), 
                                        pipes[pipe_ind].x - bird.x))
            if output[0] > 0.5:
                bird.jump()

        bird.move()

        # Move and remove pipes
        for pipe in pipes:
            pipe.move()

        if pipes and pipes[0].x + PIPE_WIDTH < 0:
            pipes.pop(0)
            score += 1

        # Add new pipe
        if frame_count % PIPE_SPAWN_INTERVAL == 0:
            pipes.append(Pipe(WIDTH))

        # Check for collisions
        if (bird.y + BIRD_RADIUS > HEIGHT or bird.y - BIRD_RADIUS < 0 or
            any(pipe.x < bird.x + BIRD_RADIUS < pipe.x + PIPE_WIDTH and
                (bird.y - BIRD_RADIUS < pipe.height or bird.y + BIRD_RADIUS > pipe.height + PIPE_GAP)
                for pipe in pipes)):
            running = False

        # Draw everything
        window.fill(WHITE)
        for pipe in pipes:
            pipe.draw()
        bird.draw()

        # Display the current score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {score}', True, BLACK)
        window.blit(score_text, (10, 10))

        pygame.display.update()

    print(f"Final Score: {score}")

def main():
    button_width = 120
    button_height = 50
    player_button = pygame.Rect(WIDTH // 4 - button_width // 2, HEIGHT // 2, button_width, button_height)
    ai_button = pygame.Rect(3 * WIDTH // 4 - button_width // 2, HEIGHT // 2, button_width, button_height)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if player_button.collidepoint(event.pos):
                    game_loop(True)
                elif ai_button.collidepoint(event.pos):
                    game_loop(False)

        window.fill(WHITE)
        draw_button(player_button, "PLAYER")
        draw_button(ai_button, "AI")
        
        title_font = pygame.font.Font(None, 48)
        title_text = title_font.render("Flappy Bird", True, BLACK)
        title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        window.blit(title_text, title_rect)

        pygame.display.update()

if __name__ == "__main__":
    main()