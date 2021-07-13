import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, file_name = "model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print ('Loading existing state dict.')
            return True
        
        print ('No existing state dict found. Starting from scratch.')
        return False

class CNN_QNet(nn.Module):
    def __init__(self, init_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(init_channels, 5 , 5)
        self.maxpool = nn.MaxPool2d(3,3)
        self.conv2 = nn.Conv2d(5, 10, 5)
        self.fc1 = nn.Linear(10*64*42 , 250)
        self.fc2 = nn.Linear(250, 80)
        self.outputl = nn.Linear(80, 3)
    
    def forward(self, x):
        print("input size: ", x.size())
        x = self.maxpool(F.relu(self.conv1(x)))
        print("conv1 out: ", x.size())
        x = self.maxpool(F.relu(self.conv2(x)))
        print("conv2 out: ", x.size())
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.outputl(x)
        print("output size: ", x.size())
        return x



    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype= torch.float)
        next_state = torch.tensor(next_state, dtype= torch.float)
        action = torch.tensor(action, dtype= torch.long)
        reward = torch.tensor(reward, dtype= torch.float)

        if len(state.shape) == 1:
            # (1 , x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over , )

        # 1: we want to get the predicted Q values for current state
        pred = self.model(state)

        # 2: reward + gamma * max( next_predicted Q value)
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


class CNN_QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype = torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype= torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype= torch.long)
        reward = torch.tensor(reward, dtype= torch.float)

        if len(state.size()) == 3:
            # (1 , x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over , )

        # 1: we want to get the predicted Q values for current state
        pred = self.model(state)

        # 2: reward + gamma * max( next_predicted Q value)
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


