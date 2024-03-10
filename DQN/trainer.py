import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from helper import plot, file_save
from agent import Agent
from game import SnakeGameAI
from datetime import datetime

from replay_buffer import ReplayBuffer

MAX_MEMORY = 100_000
BATCH_SIZE = 64
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

            self.replay_buffer = ReplayBuffer(MAX_MEMORY)

        def train_step(self, state, action, reward, next_state, done):
            states = torch.tensor(state, dtype=torch.float)
            next_states = torch.tensor(next_state, dtype=torch.float)
            actions = torch.tensor(action, dtype=torch.long)
            rewards = torch.tensor(reward, dtype=torch.float)
            dones = torch.tensor(done, dtype=torch.bool)
            # (n, x)

            # 1: predicted Q values with current state
            pred = self.model(states)

            print(pred.shape)
            print(actions.unsqueeze(-1).shape)

            #pred = torch.gather(pred, 1, actions.unsqueeze(-1)).squeeze(-1)

            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            with torch.no_grad():
                next_pred = self.model(next_states)
                next_pred_max = torch.max(next_pred, dim=1)
                print(next_pred.shape)
                target = rewards + self.gamma * next_pred_max * (~dones)
                print(target.shape)

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

                    # remember
                    self.agent.remember(state_old, final_move, reward, state_new, done)

                    # append new transition to replay buffer
                    self.replay_buffer.add(state_old, final_move, reward, state_new, done)

                    if len(self.replay_buffer.buffer) > BATCH_SIZE:
                        sample_states, sample_actions, sample_rewards, sample_next_states, sample_done = self.replay_buffer.sample(BATCH_SIZE)
                        self.train_step(sample_states, sample_actions, sample_rewards, sample_next_states, sample_done)

                    if program_end:
                        raise Exception

                    if done:
                        # train long memory, plot result
                        self.env.reset()
                        self.agent.n_games += 1

                        # basic q-learning part..
                        #self.train_long_memory()

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
                print(e)
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
