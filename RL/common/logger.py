import numpy as np
import pandas as pd
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.utils import make_grid
import time
import yaml
import os

class Logger(object):
    
    def __init__(self, n_envs, logdir, config_dict):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir

        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
        self.episode_len_buffer = deque(maxlen = 40)
        self.episode_reward_buffer = deque(maxlen = 40)
        
        self.log = pd.DataFrame(columns = ['timesteps', 'wall_time', 'num_episodes',
                               'max_episode_rewards', 'mean_episode_rewards','min_episode_rewards',
                               'max_episode_len', 'mean_episode_len', 'min_episode_len', 'normalized_test_mean_episodes', 'test_mean_episodes'])
        self.writer = SummaryWriter(logdir)
        self.timesteps = 0
        self.num_episodes = 0
        self.count = 0
        
        json_file_path = os.path.join(logdir, 'config.yaml')   
        with open(json_file_path, 'w') as f:
            documents = yaml.dump(config_dict, f) 
        
        
        
    def feed(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
        self.timesteps += (self.n_envs * steps)

    def write_summary(self, summary):
        for key, value in summary.items():
            self.writer.add_scalar(key, value, self.timesteps)
    
    def write_test_summary(self, test_mean_reward):
        self.writer.add_scalar('test mean reward', test_mean_reward, self.timesteps)

    def dump(self, normalized_test_log=None, test_log=None):
        wall_time = time.time() - self.start_time
        if self.num_episodes > 0:
            episode_statistics = self._get_episode_statistics()
            episode_statistics['Rewards/ normalized_test_mean_episodes'] = normalized_test_log
            episode_statistics['Rewards/ test_mean_episodes'] = test_log
            episode_statistics_list = list(episode_statistics.values())
            for key, value in episode_statistics.items():
                self.writer.add_scalar(key, value, self.timesteps)
        else:
            episode_statistics_list = [None] * 8
        log = [self.timesteps] + [wall_time] + [self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        # TODO: logger to append, not write!
        with open(self.logdir + '/log.csv', 'w') as f:
            self.log.to_csv(f, index = False)
        print(self.log.loc[len(self.log)-1])

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = np.max(self.episode_reward_buffer)
        episode_statistics['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes']  = np.min(self.episode_reward_buffer)
        episode_statistics['Len/max_episodes']  = np.max(self.episode_len_buffer)
        episode_statistics['Len/mean_episodes'] = np.mean(self.episode_len_buffer)
        episode_statistics['Len/min_episodes']  = np.min(self.episode_len_buffer)
        return episode_statistics

    def log_images(self, imgs_originals, imgs_prime):
        img_log = self._get_decoded_images(imgs_originals, 2)
        self.writer.add_image('originals', img_log, global_step=self.count)
        img_log = self._get_decoded_images(imgs_prime, 2)
        self.writer.add_image('mix', img_log, global_step=self.count)
        self.count += 1

    def _get_decoded_images(self, images, nrow=4):
        n = len(images)
        im = []
        for image in images:
            im.append(make_grid(image, nrow=nrow))
            
        return torch.cat(im, dim=2)