from TileCoder.TileCoderSutton import *
from DQN.NeuralNetwork import *

GAMMA = 0.995

class FunctionApproximator:
    def __init__(self, params={}):
        self.type = None

        # tile coder parameters
        self.weights = None
        self.iht_size = None
        self.initial_weights = None
        self.num_tiles = None
        self.num_tilings = None
        self.tile_coder = None
        self.num_actions = None

        # neural network parameters
        self.target_net = None
        self.eval_net = None
        self.update_target_counter = 0
        self.memory_size = None
        self.memory = None
        self.memory_counter = 0
        self.loss_func = nn.MSELoss()
        self.update_target_rate = None
        self.batch_size = None

        self.set_params_from_dict(params)

    def set_params_from_dict(self, params={}):
        self.type = params.get("type", "tile coder")

        self.initialize_function_approximator(params)

    def initialize_tile_coder(self, params = {}):
        self.iht_size = params.get("iht_size", 4096)
        self.initial_weights = params.get("initial_weights", 0.0)
        self.num_tiles = params.get("num_tiles", 8)
        self.num_tilings = params.get("num_tilings", 8)
        self.num_actions = params.get("num_actions", 3)
        self.weights = np.ones((self.num_actions, self.iht_size)) * self.initial_weights

        self.tile_coder = TileCoderSutton({
            'max_size': self.iht_size,
            'num_tilings': self.num_tilings,
            'num_tiles': self.num_tiles,
            'min_values': params.get("env_min_values", 0.0),
            'max_values': params.get("env_max_values", 0.0)
        })

    def initialize_neural_network(self, params = {}):
        self.state_dim = params.get("state_dim", 4)
        self.action_dim = params.get("action_dim", 2)
        self.target_net, self.eval_net = Net(self.state_dim, self.action_dim), Net(self.state_dim, self.action_dim)
        self.update_target_counter = params.get("update_counter", 0)
        self.memory_size = params.get("memory_size", 200)
        self.update_target_rate = params.get("learn_every", 50)
        self.memory = np.zeros((self.memory_size, 2 * self.state_dim + 2))
        self.batch_size = params.get("batch_size", 128)

    def initialize_function_approximator(self, params={}):
        if self.type == "tile coder":
            self.initialize_tile_coder(params)
        elif self.type == "neural network":
            self.initialize_neural_network(params)

    def get_action_value(self, state, action=None):
        # TODO: tune no_grad later
        action_value = None
        if self.type == "tile coder":
            action_value = np.sum(self.get_weights(state, action), axis=-1)
        elif self.type == "neural network":
            if action is None:
                action_value = self.eval_net(state)
            else:
                action_value = self.eval_net(state)[action]

        return action_value

    def get_weights(self, state, action=None):
        tiles = self.tile_coder.get_activated_tiles(state)
        if action is not None:
            weights = self.weights[action, tiles]
        else:
            weights = self.weights[:, tiles]
        return weights

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def change_weights(self, learning_rate, delta, state, action):
        if self.type == "tile coder":
            grad = np.ones(self.num_tilings)
            tiles = self.tile_coder.get_activated_tiles(state)
            self.weights[action, tiles] += (learning_rate / self.num_tilings) * delta * grad

        if self.type == "neural network":

            # every n learning cycle, the target network will be replaced with the eval network
            if self.update_target_counter % self.update_target_rate == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.update_target_counter += 1

            # we can start learning when the memory is full
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, self.batch_size)
                batch_memory = self.memory[sample_index, :]
                #print(batch_memory)
                batch_state = torch.FloatTensor(batch_memory[:, :self.state_dim])
                batch_action = torch.LongTensor(batch_memory[:, self.state_dim:self.state_dim + 1].astype(int))
                batch_reward = torch.FloatTensor(batch_memory[:, self.state_dim + 1:self.state_dim + 2])
                batch_next_state = torch.FloatTensor(batch_memory[:, -self.state_dim:])

                q_eval = self.eval_net(batch_state).gather(1, batch_action)
                q_next = self.target_net(batch_next_state).detach()
                q_target = batch_reward + GAMMA * q_next.max(1)[0].view(self.batch_size, 1) # TODO: change that
                loss = self.loss_func(q_eval, q_target)
                self.eval_net.backpropagate(loss)
