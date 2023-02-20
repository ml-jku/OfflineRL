import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .agent import Agent
from ..utils.evaluation import entropy
from ..networks.critic import Critic
from ..networks.actor import Actor


class BCQ(Agent):
    """
    Re-implementation of the original author implementation found at
    https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 discount,
                 lr=1e-4,
                 seed=None):
        super(BCQ, self).__init__(obs_space, action_space, discount, lr, seed)

        # epsilon decay
        self.initial_eps = 1.0
        self.end_eps = 1e-2
        self.eps_decay_period = 1000
        self.slope = (self.end_eps - self.initial_eps) / self.eps_decay_period
        self.eval_eps = 0.

        # loss function
        self.huber = nn.SmoothL1Loss()
        self.nll = nn.NLLLoss()

        # threshold for actions, unlikely under the behavioral policy
        self.threshold = 0.3

        # Number of training iterations
        self.iterations = 0

        # After how many training steps 'snap' target to main network?
        self.target_update_freq = 100

        # Q-Networks
        self.Q = Critic(self.obs_space, self.action_space, seed).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        # BCQ has a separate Actor, as I have no common vision network
        self.actor = Actor(self.obs_space, self.action_space, seed).to(self.device)

        # Optimization
        self.optimizer = torch.optim.Adam(params=list(self.Q.parameters())+list(self.actor.parameters()), lr=self.lr)

    def policy(self, state, eval=False):

        # set networks to eval mode
        self.actor.eval()
        self.Q.eval()

        if eval:
            eps = self.eval_eps
        else:
            eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # epsilon greedy policy
        if self.rng.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)

                q_val = self.Q(state).cpu()
                actions = self.actor(state).cpu()

                sm = F.log_softmax(actions, dim=1).exp()
                mask = ((sm / sm.max(1, keepdim=True)[0]) > self.threshold).float()

                # masking non-eligible values with -9e9 to be sure they are not sampled
                return int((mask * q_val + (1. - mask) * -9e9).argmax(dim=1)), \
                       q_val, entropy(sm)
        else:
            return self.rng.integers(self.action_space), np.nan, np.nan

    def train(self, buffer, writer, minimum=None, maximum=None, use_probas=False):
        # Sample replay buffer
        state, action, next_state, reward, not_done = buffer.sample(minimum, maximum, use_probas)

        # set networks to train mode
        self.actor.train()
        self.Q.train()
        self.Q_target.train()

        with torch.no_grad():
            q_val = self.Q(next_state)
            actions = self.actor(next_state)

            sm = F.log_softmax(actions, dim=1).exp()
            sm = (sm / sm.max(1, keepdim=True)[0] > self.threshold).float()
            next_action = (sm * q_val + (1. - sm) * -9e9).argmax(dim=1, keepdim=True)

            q_val = self.Q_target(next_state)
            target_Q = reward + not_done * self.discount * q_val.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate and actor decisions on actions
        current_Q = self.Q(state).gather(1, action)
        actions = self.actor(state)

        # Compute Q loss (Huber loss)
        Q_loss = self.huber(current_Q, target_Q)
        A_loss = self.nll(F.log_softmax(actions, dim=1), action.reshape(-1))
        # third term is
        loss = Q_loss + A_loss + 1e-2 * actions.pow(2).mean()

        # log temporal difference error
        if self.iterations % 100 == 0:
            writer.add_scalar("train/TD-error", torch.mean(Q_loss).detach().cpu().item(), self.iterations)
            writer.add_scalar("train/CE-loss", torch.mean(A_loss).detach().cpu().item(), self.iterations)

        # Optimize the Q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iterations += 1
        # Update target network by full copy every X iterations.
        if self.iterations % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def get_name(self) -> str:
        return "BatchConstrainedQLearning"

    def save_state(self) -> None:
        torch.save(self.Q.state_dict(), os.path.join("models", self.get_name() + "_Q.pt"))
        torch.save(self.actor.state_dict(), os.path.join("models", self.get_name() + "_actor.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", self.get_name() + "_optim.pt"))

    def load_state(self) -> None:
        self.Q.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_Q.pt")))
        self.Q_target = copy.deepcopy(self.Q)
        self.actor.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_actor.pt")))
        self.optimizer.load_state_dict(torch.load(os.path.join("models", self.get_name() + "_optim.pt")))