from gymenv_v2 import make_multiple_env
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import wandb
wandb.login()
run=wandb.init(project="testproject", entity="vrp", tags=["test"])

### TRAINING

# Setup: You may generate your own instances on which you train the cutting agent.
our_config = {
    "load_dir"        : 'instances/randomip_n2_m6',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(50)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}


test_config = {
    "load_dir" : 'instances/test_100_n60_m60',
    "idx_list" : list(range(99)),
    "timelimit" : 50,
    "reward_type" : 'obj'
}


class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(LSTM_net, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, input):
        hidden = self.init_hidden()
        inputs = torch.FloatTensor(input).view(1, -1, self.input_size)
        output, _ = self.lstm(inputs)
        # output[-1] is same as last hidden state
        output = output[-1].reshape(-1, self.hidden_size)
        return output

    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))


class Attention_Net(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(Attention_Net, self).__init__()
        # constrain and cuts dimension
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.hidden_size2 = int(hidden_size2)
        self.lstm1 = LSTM_net(input_size, hidden_size)
        self.lstm2 = LSTM_net(input_size, hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size2)
        self.linear2 = nn.Linear(self.hidden_size2, self.hidden_size2)
        self.tanh = nn.Tanh()

    def forward(self, constraints, cuts):
        constraints = torch.FloatTensor(constraints)
        cuts = torch.FloatTensor(cuts)

        # lstm
        A_embed = self.lstm1.forward(constraints)
        D_embed = self.lstm2.forward(cuts)

        # dense
        A = self.linear2(self.tanh(self.linear1(A_embed)))
        D = self.linear2(self.tanh(self.linear1(D_embed)))

        # attention
        logits = torch.sum(torch.mm(D, A.T), axis=1)

        return logits


# Policy network will just be copied from lab4 and make small modification
class Policy(object):
    def __init__(self, input_size, hidden_size, hidden_size2, lr):

        self.model = Attention_Net(input_size, hidden_size, hidden_size2)
        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_prob(self, constraints, cuts):
        constraints = torch.FloatTensor(constraints)
        cuts = torch.FloatTensor(cuts)
        prob = torch.nn.functional.softmax(self.model(constraints, cuts), dim=-1)
        return prob.cpu().data.numpy()

    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(self, constraints, cuts, actions, Qs):
        """
        states: numpy array (states)
        actions: numpy array (actions)
        Qs: numpy array (Q values)
        """
        actions = torch.LongTensor(actions)
        Qs = torch.FloatTensor(Qs)

        total_loss = 0
        # for a bunch of constraints and cuts, need to go one by one
        for i in range(len(constraints)):
            curr_constraints = constraints[i]
            curr_cuts = cuts[i]
            curr_action = actions[i]
            # COMPUTE probability vector pi(s) for all s in states
            logits = self.model(curr_constraints, curr_cuts)
            prob = torch.nn.functional.softmax(logits, dim=-1)
            # Compute probaility pi(s,a) for all s,a
            action_onehot = self._to_one_hot(curr_action, curr_cuts.shape[0])
            prob_selected = torch.sum(prob * action_onehot, axis=-1)

            # FOR ROBUSTNESS
            prob_selected += 1e-8
            loss = -torch.mean(Qs[i] * torch.log(prob_selected))
            # BACKWARD PASS
            self.optimizer.zero_grad()
            loss.backward()
            # UPDATE
            self.optimizer.step()
            total_loss += loss.detach().cpu().data.numpy()

        return total_loss


def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


def normalization(A, b, E, d):
    all_coeff = np.concatenate((A, E), axis=0)
    all_constraint = np.concatenate((b, d))
    max_1, max_2 = np.max(all_coeff), np.max(all_constraint)
    min_1, min_2 = np.min(all_coeff), np.min(all_constraint)
    norm_A = (A - min_1) / (max_1 - min_1)
    norm_E = (E - min_1) / (max_1 - min_1)
    norm_b = (b - min_2) / (max_2 - min_2)

    norm_d = (d - min_2) / (max_2 - min_2)

    return norm_A, norm_b, norm_E, norm_d


if __name__ == "__main__":

    # training = False
    training = True
    explore = True
    PATH = "models/Qiming/our_config_best_model_2_6.pt"
    # PATH = "models/hard_config_best_model3.pt"

    # create env
    # env = make_multiple_env(**test_config)
    # env = make_multiple_env(**custom_config)
    env = make_multiple_env(**our_config)
    lr = 1e-2
    # initialize networks
    # input_dim = 61
    input_dim = 3
    lstm_hidden = 10
    # lstm_hidden = 5
    dense_hidden = 64

    explore_rate = 1.0
    min_explore_rate = 0.01
    max_explore_rate = 0.5
    explore_decay_rate = 0.01
    best_rew = 0

    if training:
        actor = Policy(input_size=input_dim, hidden_size=lstm_hidden, hidden_size2=dense_hidden, lr=lr)
    else:
        actor = torch.load(PATH)

    sigma = 0.2
    gamma = 0.99 # discount
    rrecord = []
    for e in range(50):
        # gym loop
        # To keep a record of states actions and reward for each episode
        obss_constraint = []  # states
        obss_cuts = []
        acts = []
        rews = []

        s = env.reset()   # samples a random instance every time env.reset() is called
        d = False
        repisode = 0
        while not d:

            A, b, c0, cuts_a, cuts_b = s

            # normalization
            A, b, cuts_a, cuts_b = normalization(A, b, cuts_a, cuts_b)

            # concatenate [a, b] [e, d]
            curr_constraints = np.concatenate((A, b[:, None]), axis=1)
            available_cuts = np.concatenate((cuts_a, cuts_b[:, None]), axis=1)

            # compute probability distribution
            prob = actor.compute_prob(curr_constraints, available_cuts)
            prob /= np.sum(prob)

            explore_rate = min_explore_rate + \
                           (max_explore_rate - min_explore_rate) * np.exp(-explore_decay_rate * (e))

            # epsilon greedy for exploration
            if training and explore:
                random_num = random.uniform(0, 1)
                if random_num <= explore_rate:
                    a = np.random.randint(0, s[-1].size, 1)
                else:
                    #a = np.argmax(prob)
                    a = [np.random.choice(s[-1].size,  p=prob.flatten())]
            else:
                # for testing case, only sample action
                a = [np.random.choice(s[-1].size,  p=prob.flatten())]

            new_state, r, d, _ = env.step(list(a))
            #print('episode', e, 'step', t, 'reward', r, 'action space size', new_state[-1].size, 'action', a)
            #a = np.random.randint(0, s[-1].size, 1) # s[-1].size shows the number of actions, i.e., cuts available at state s
            #A, b, c0, cuts_a, cuts_b = new_state

            obss_constraint.append(curr_constraints)
            obss_cuts.append(available_cuts)
            acts.append(a)
            rews.append(r)
            s = new_state
            repisode += r

        # record rewards and print out to track performance
        rrecord.append(np.sum(rews))
        returns = discounted_rewards(rews, gamma)
        # we only use one trajectory so only one-variate gaussian used here.
        Js = returns + np.random.normal(0, 1, len(returns)) / sigma
        print("episode: ", e)
        print("sum reward: ", repisode)

        # PG update and save best model so far
        if training:
            if repisode >= best_rew:
                best_rew = repisode
                torch.save(actor, PATH)

            loss = actor.train(obss_constraint, obss_cuts, acts, Js)
            print("Loss: ", loss)


        #wandb logging
        wandb.log({"Discounted Reward": np.sum(returns)})
        fixedWindow = 10
        movingAverage = 0
        if len(rrecord) >= fixedWindow:
            movingAverage = np.mean(rrecord[len(rrecord) - fixedWindow:len(rrecord) - 1])
        wandb.log({"Training reward": repisode, "training reward moving average": movingAverage})


	#if using hard-config make sure to use "training-hard" tag in wandb.init in the initialization on top
