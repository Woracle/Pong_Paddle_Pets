import torch
import random
import numpy as np
from collections import deque
from pongAI import PongGameAI
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # controls random
        self.gamma = 0.9 # discount rate
        self.L_memory = deque(maxlen = MAX_MEMORY) # once it gets full it removes early events
        self.R_memory = deque(maxlen = MAX_MEMORY) # once it gets full it removes early events
        self.modelp1 = Linear_QNet(8, 256, 3)
        self.modelp2 = Linear_QNet(8, 256, 3)
        self.L_trainer = QTrainer(self.modelp1, lr = LR, gamma = self.gamma)
        self.R_trainer = QTrainer(self.modelp2, lr = LR, gamma = self.gamma)

    def get_state(self, game):

        ball_x = game.ball_pos[0]
        ball_y = game.ball_pos[1]
        ball_x_vel = game.ball_vel[0]
        ball_y_vel = game.ball_vel[1]

        paddle1_pos = game.paddle1_pos[1]
        paddle1_vel = game.paddle1_vel
        paddle2_pos = game.paddle2_pos[1]
        paddle2_vel = game.paddle2_vel

        # so state will be:
        # 1. is the ball above the paddle bool
        if paddle1_pos < ball_y:
            BAP1 = 1
        else: 
            BAP1 = 0
        if paddle2_pos < ball_y:
            BAP2 = 1
        else: 
            BAP2 = 0
        # 2. is the ball below the paddle bool
        if paddle1_pos > ball_y:
            BBP1 = 1
        else: 
            BBP1 = 0
        if paddle2_pos > ball_y:
            BBP2 = 1
        else: 
            BBP2 = 0
        # is the ball moving towards the paddle
        if ball_x_vel < 0:
            BTP1 = 1
        else:
            BTP1 = 0
        if ball_x_vel > 0:
            BTP2 = 1
        else:
            BTP2 = 0  
        # is the ball moving up
        if ball_y_vel > 0:
            BVU = 1
        else:
            BVU = 0
        # is the ball moving down
        if ball_y_vel < 0:
            BVD = 1
        else:
            BVD = 0
        # is the opponants paddle above or below your paddle
        if paddle1_pos > paddle2_pos:
            P2P1 = 0
            P2P2 = 1
        elif paddle1_pos < paddle2_pos:
            P2P1 = 1
            P2P2 = 0
        else:
            P2P1 = 0
            P2P2 = 0
        # the oppenants paddle moving upwards or downwards
        if paddle1_vel > 0:
            P1U = 1
        else: 
            P1U = 0

        if paddle1_vel <= 0:
            P1D = 1
        else: 
            P1D = 0

        if paddle2_vel > 0:
            P2U = 1
        else: 
            P2U = 0

        if paddle2_vel <= 0:
            P2D = 1
        else: 
            P2D = 0

        L_state = np.array([BAP1, BBP1, BTP1, BVU, BVD, P2P1, P2U, P2D], dtype= int)
        R_state = np.array([BAP2, BBP2, BTP2, BVU, BVD, P2P2, P1U, P1D], dtype= int)

        return L_state, R_state
        

    def remember(self, state_L, state_R, Left_Move, Right_Move, L_reward, R_reward, state_L_New, state_R_New, game_over):
        self.L_memory.append((state_L, Left_Move, L_reward,state_L_New, game_over ))
        self.R_memory.append((state_R, Right_Move, R_reward,state_R_New, game_over ))

    def train_long_memory(self):
        if len(self.L_memory) > BATCH_SIZE:
            L_mini_sample = random.sample(self.L_memory, BATCH_SIZE)
        else:
            L_mini_sample = self.L_memory

        states, actions, rewards, next_states, game_overs = zip(*L_mini_sample)
        
        self.L_trainer.train_step(states, actions, rewards, next_states, game_overs)

        if len(self.R_memory) > BATCH_SIZE:
            R_mini_sample = random.sample(self.R_memory, BATCH_SIZE)
        else:
            R_mini_sample = self.R_memory

        states, actions, rewards, next_states, game_overs = zip(*R_mini_sample)
        
        self.R_trainer.train_step(states, actions, rewards, next_states, game_overs)
        
        
  
    def train_short_memory(self, state_L, state_R, Left_Move, Right_Move, L_reward, R_reward, state_L_New, state_R_New, game_over):
        self.L_trainer.train_step(state_L, Left_Move, L_reward,state_L_New, game_over)
        self.R_trainer.train_step(state_R, Right_Move, R_reward,state_R_New, game_over)

    def get_action(self, state_L, state_R):
        self.epsilon = 30 - self.n_games
        Left_Move = [0,0,0]
        Right_Move = [0,0,0]

        if random.randint(0,200) < self.n_games:
            moveL = random.randint(0,2)
            moveR = random.randint(0,2)
            Left_Move[moveL] = 1
            Right_Move[moveR] = 1
        else: 
            stateL = torch.tensor(state_L, dtype = torch.float)
            stateR = torch.tensor(state_R, dtype = torch.float)
            L_pred = self.modelp1(stateL)
            R_pred = self.modelp2(stateR)
            moveL = torch.argmax(L_pred).item()
            moveR = torch.argmax(R_pred).item()
            Left_Move[moveL] = 1
            Right_Move[moveR] = 1

        return Left_Move, Right_Move


def train():
    game = PongGameAI()
    agent = Agent()
    best_L_Reward = 0 
    best_R_Reward = 0
    total_L_Reward = 0
    total_R_reward = 0
    winner_hist = []
        
    while True:
        state_L, state_R = agent.get_state(game)

        Left_Move, Right_Move = agent.get_action(state_L, state_R)

        L_reward, R_reward, game_over, winner, l_score, r_score = game.play_step(Left_Move, Right_Move)
        state_L_New, state_R_New = agent.get_state(game)

        total_L_Reward += L_reward
        total_R_reward += R_reward
        winner_hist.append(winner)

        # train short memory
        agent.train_short_memory(state_L, state_R, Left_Move, Right_Move, L_reward, R_reward, state_L_New, state_R_New, game_over)

        # remember
        agent.remember(state_L, state_R, Left_Move, Right_Move, L_reward, R_reward, state_L_New, state_R_New, game_over)

        if game_over:
            # train the long_memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if L_reward > best_L_Reward:
                best_L_Reward = L_reward
                agent.modelp1.save(file_name= "Left_Paddle.pth")

            if R_reward > best_R_Reward:
                best_R_Reward = R_reward
                agent.modelp2.save(file_name= "Right_Paddle.pth")  

            # TODO impliment a score / winner tracker
            print(f'Left paddle total reward {total_L_Reward} \n Right paddle total reward {total_R_reward} \n Games won Left {sum(np.array(winner_hist)=="LEFT")} \n Games won right {sum(np.array(winner_hist)=="RIGHT")} \nScore Left {l_score} \n Score Right {r_score}')         





if __name__ == '__main__':
    train()

