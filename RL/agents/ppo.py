from .base_agent import BaseAgent
from common.augmentations import *
from common.misc_util import adjust_lr
import torch
import torch.optim as optim
import numpy as np

numpy_aug_to_func = {    
                'cutout':random_cutout,
                'cutout_color':random_cutout_color,
                'crop':random_crop,
                'no_aug': no_aug
                }

torch_aug_to_func = {    
        'gray':random_grayscale,
        'flip':random_flip,
        'rotate':random_rotation,
        'random_conv':random_convolution,
        'no_aug': no_aug
        }


class PPO(BaseAgent):
    def __init__(self,
                 env,
                 test_env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=0.0005,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 numpy_augmentation='no_aug',
                 pytorch_augmentation='no_aug',
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device, n_checkpoints)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae
        self.test_env = test_env
        self.numpy_augmentation = numpy_aug_to_func[numpy_augmentation]
        self.pytorch_augmentation = torch_aug_to_func[pytorch_augmentation]
        self.test_limit = 0

    def predict(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            augmented_obs = self.pytorch_augmentation(obs)
            dist, value = self.policy(augmented_obs)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)
        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size)
            for sample in generator:
                obs_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                augmented_obs = self.pytorch_augmentation(obs_batch.to(self.device))
                dist_batch, value_batch = self.policy(augmented_obs)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(pi_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def test(self):
        self.policy.eval()
        obs = self.test_env.reset()
        episode = 0
        normalized_total_rewards = []
        total_rewards = []
        with torch.no_grad():
            while True:
                act, _, _ = self.predict(obs)
                obs, rew, dones, infos = self.test_env.step(act)
                real_rew = [info['env_reward'] for info in infos]
                normalized_total_rewards.append(sum(rew))
                total_rewards.append(sum(real_rew))
                if sum(dones) > 0:
                    episode += sum(dones)
                if episode >= 1000:
                    break
        #TODO make it cleaner
        return sum(normalized_total_rewards)/episode, sum(total_rewards)/episode

    def train(self, num_timesteps):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        done = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.train()
            for _ in range(self.n_steps):
                obs = self.numpy_augmentation(obs)
                act, log_prob_act, value = self.predict(obs)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, act, rew, done, info, log_prob_act, value)
                obs = next_obs
            _, _, last_val = self.predict(obs)
            self.storage.store_last(obs, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)
            
            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            self.logger.feed(rew_batch, done_batch)
            self.logger.write_summary(summary)
            if self.t > self.test_limit:
                normalized_mean_reward, test_mean_rewards = self.test()
                self.logger.write_test_summary(normalized_mean_reward)
                self.logger.dump(normalized_mean_reward, test_mean_rewards)
                self.test_limit += 1000000
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt+1) * save_every):
                torch.save({'state_dict': self.policy.state_dict()}, self.logger.logdir +
                           '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()
