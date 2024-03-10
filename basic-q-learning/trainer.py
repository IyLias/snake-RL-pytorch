import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from helper import plot, file_save
from agent import Agent
from game import SnakeGameAI
from datetime import datetime

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class QTrainer:
        def __init__(self, agent, env, lr, gamma):
            self.lr = lr
            self.gamma = gamma

            self.agent = agent
            self.env = env
            self.model = self.agent.model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()

        def train_step(self, state, action, reward, next_state, done):
            state = torch.tensor(state, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            # (n, x)

            if len(state.shape) == 1:
                # (1, x)
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                done = (done,)

            # 1: predicted Q values with current state
            pred = self.model(state)

            target = pred.clone()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

                target[idx][torch.argmax(action[idx]).item()] = Q_new

            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
            self.optimizer.zero_grad()
            loss = self.criterion(target, pred)
            loss.backward()

            self.optimizer.step()

        def train(self):
            plot_scores = []
            plot_mean_scores = []
            total_score = 0
            record = 0

            try:
                while True:
                    # get old state
                    state_old = self.agent.get_state(self.env)

                    # get move
                    final_move = self.agent.get_action(state_old)

                    # perform move and get new state
                    reward, done, score, program_end = self.env.play_step(final_move)
                    state_new = self.agent.get_state(self.env)

                    # train short memory
                    self.train_short_memory(state_old, final_move, reward, state_new, done)

                    # remember
                    self.agent.remember(state_old, final_move, reward, state_new, done)

                    if program_end:
                        raise Exception

                    if done:
                        # train long memory, plot result
                        self.env.reset()
                        self.agent.n_games += 1
                        self.train_long_memory()

                        if score > record:
                            record = score
                            self.agent.model.save()

                        print('Game', self.agent.n_games, 'Score', score, 'Record:', record)

                        plot_scores.append(score)
                        total_score += score
                        mean_score = total_score / self.agent.n_games
                        plot_mean_scores.append(mean_score)
                        plot(plot_scores, plot_mean_scores)

            except KeyboardInterrupt:
                print("Exit by KeyBoard Interrupt. Program Ended")
                file_save(datetime.now(), plot_scores, plot_mean_scores)
                quit()

            except Exception as e:
                print("Program Ended")
                file_save(datetime.now(), plot_scores, plot_mean_scores)
                quit()



        def train_long_memory(self):
            if len(self.agent.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.agent.memory, BATCH_SIZE)  # list of tuples
            else:
                mini_sample = self.agent.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.train_step(states, actions, rewards, next_states, dones)
            # for state, action, reward, nexrt_state, done in mini_sample:
            #    self.trainer.train_step(state, action, reward, next_state, done)

        def train_short_memory(self, state, action, reward, next_state, done):
            self.train_step(state, action, reward, next_state, done)



if __name__ == '__main__':

    agent = Agent()
    game = SnakeGameAI()

    R = QTrainer(agent, game, lr=LR, gamma = agent.gamma)
    R.train()
