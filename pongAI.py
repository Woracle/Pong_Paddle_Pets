import random
import pygame
import numpy as np


pygame.init()

#colors
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)

SPEED = 200
      

class PongGameAI:

    def __init__(self, w = 600, h = 400):
        self.w = w
        self.h = h
        self.BALL_RADIUS = 20
        self.PAD_WIDTH = 8
        self.PAD_HEIGHT = 80
        self.HALF_PAD_WIDTH = self.PAD_WIDTH // 2
        self.HALF_PAD_HEIGHT = self.PAD_HEIGHT // 2
        self.ball_pos = [0,0]
        self.ball_vel = [0,0]
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

               # init game state
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1,self.h//2]
        self.paddle2_pos = [self.w +1 - self.HALF_PAD_WIDTH, self.h//2]
        self.l_score = 0
        self.r_score = 0
        self.L_reward = 0
        self.R_reward = 0
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.ball_pos = [self.w//2,self.h//2]
        self.ball_vel = [random.randrange(2,4), random.randrange(1,3)]
        self.frame_iteration = 0

        self.display.fill(BLACK)
        pygame.draw.line(self.display, WHITE, [self.w // 2, 0],[self.w // 2, self.h], 1)
        pygame.draw.line(self.display, WHITE, [self.PAD_WIDTH, 0],[self.PAD_WIDTH, self.h], 1)
        pygame.draw.line(self.display, WHITE, [self.w - self.PAD_WIDTH, 0],[self.w - self.PAD_WIDTH, self.h], 1)
        pygame.draw.circle(self.display, WHITE, [self.w//2, self.h//2], 70, 1)

         #draw paddles and ball
        pygame.draw.circle(self.display, RED, self.ball_pos, 20, 0)
        pygame.draw.polygon(self.display, GREEN, [[self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(self.display, BLUE, [[self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT]], 0)




    def ball_init(self, right):
        self.ball_pos = [self.w//2,self.h//2]
        horz = random.randrange(2,4)
        vert = random.randrange(1,3)
    
        if right == False:
            horz = - horz
        
        self.ball_vel = [horz,-vert]

    def reset(self):
        self.l_score = 0
        self.r_score = 0
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1,self.h//2]
        self.paddle2_pos = [self.w +1 - self.HALF_PAD_WIDTH,self.h//2]        
        if random.randrange(0,2) == 0:
            self.ball_init(True)
        else:
            self.ball_init(False)
        self.L_reward = 0
        self.R_reward = 0
        self.frame_iteration = 0
    
    def is_collision(self):

        # hits the paddles
        if int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH and int(self.ball_pos[1]) in range(self.paddle1_pos[1] - self.HALF_PAD_HEIGHT, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT,1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
            return "pl"


        if int(self.ball_pos[0]) >= self.w + 1 - self.BALL_RADIUS - self.PAD_WIDTH and int(self.ball_pos[1]) in range(self.paddle2_pos[1] - self.HALF_PAD_HEIGHT, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT,1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
            return "pr"

        # hits the wall

        if int(self.ball_pos[0]) - self.BALL_RADIUS <= 0:
            self.r_score += 1
            self.ball_init(False)
            return "wl"

        
        if int(self.ball_pos[0]) + self.BALL_RADIUS >= self.w:
            self.l_score += 1
            self.ball_init(True)
            return "wr"


        # hits top or bottom
        if int(self.ball_pos[1]) <= self.BALL_RADIUS:
            self.ball_vel[1] = - self.ball_vel[1]
        if int(self.ball_pos[1]) >= self.h + 1 - self.BALL_RADIUS:
            self.ball_vel[1] = -self.ball_vel[1]

    def move(self, paddle1Action, paddle2Action):

        # adjust velocity 
        if np.array_equal(paddle1Action, [1,0,0]):
            self.paddle1_vel = 8
        elif np.array_equal(paddle1Action, [0,0,1]):
            self.paddle1_vel = -8
        else:
            self.paddle1_vel = 0

        if np.array_equal(paddle2Action, [1,0,0]):
            self.paddle2_vel = 8
        elif np.array_equal(paddle2Action, [0,0,1]):
            self.paddle2_vel = -8
        else:
            self.paddle2_vel = 0

        # move the paddles
        if self.paddle1_pos[1] > self.HALF_PAD_HEIGHT and self.paddle1_pos[1] < self.h - self.HALF_PAD_HEIGHT:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HALF_PAD_HEIGHT and self.paddle1_vel > 0:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.h - self.HALF_PAD_HEIGHT and self.paddle1_vel < 0:
            self.paddle1_pos[1] += self.paddle1_vel
    
        if self.paddle2_pos[1] > self.HALF_PAD_HEIGHT and self.paddle2_pos[1] < self.h - self.HALF_PAD_HEIGHT:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HALF_PAD_HEIGHT and self.paddle2_vel > 0:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.h - self.HALF_PAD_HEIGHT and self.paddle2_vel < 0:
            self.paddle2_pos[1] += self.paddle2_vel


        # move the ball
        self.ball_pos[0] += int(self.ball_vel[0])
        self.ball_pos[1] += int(self.ball_vel[1])
    
    def update_ui(self):
        self.display.fill(BLACK)
        pygame.draw.line(self.display, WHITE, [self.w // 2, 0],[self.w // 2, self.h], 1)
        pygame.draw.line(self.display, WHITE, [self.PAD_WIDTH, 0],[self.PAD_WIDTH, self.h], 1)
        pygame.draw.line(self.display, WHITE, [self.w - self.PAD_WIDTH, 0],[self.w - self.PAD_WIDTH, self.h], 1)
        pygame.draw.circle(self.display, WHITE, [self.w//2, self.h//2], 70, 1)

         #draw paddles and ball
        pygame.draw.circle(self.display, RED, self.ball_pos, 20, 0)
        pygame.draw.polygon(self.display, GREEN, [[self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT]], 0)
        pygame.draw.polygon(self.display, BLUE, [[self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT], [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT]], 0)        
        
        pygame.display.flip()
    
    def play_step(self, paddle1Action, paddle2Action):
        self.frame_iteration += 1

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # move paddles and check collisions
        self.move(paddle1Action, paddle2Action)

        coll = self.is_collision()
        
        if coll == None:
            pass
        elif coll == "pl" or coll == "wr":
            self.L_reward += 5 
        elif coll == "pr" or coll == "wl":
            self.R_reward += 5

        # check if the game is over
        game_over = False
        winner = None
        if self.l_score >= 5:
            game_over = True
            self.L_reward += 10
            self.R_reward += -10
            winner = "LEFT"
            return self.L_reward, self.R_reward, game_over, winner, self.l_score, self.r_score 
        
        if self.r_score >= 5:
            game_over = True
            self.L_reward += -10
            self.R_reward += 10
            winner = "RIGHT"
            return self.L_reward, self.R_reward, game_over, winner, self.l_score, self.r_score

        # if not over update the UI
        self.update_ui()
        self.clock.tick(SPEED)

        # return rewards, scores and game_over status
        return self.L_reward, self.R_reward, game_over, winner, self.l_score, self.r_score    
