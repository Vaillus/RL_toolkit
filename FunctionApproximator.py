from TileCoder.TileCoderSutton import *


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

        self.eligibility_traces = None
        self.trace_decay = None

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

    def initialize_function_approximator(self, params={}):
        if self.type == "tile coder":
            self.initialize_tile_coder(params)

    def get_action_value(self, state, action=None):
        action_value = None
        if self.type == "tile coder":
            action_value = np.sum(self.get_weights(state, action), axis=-1)

        return action_value

    def get_weights(self, state, action=None):
        tiles = self.tile_coder.get_activated_tiles(state)
        if action is not None:
            weights = self.weights[action, tiles]
        else:
            weights = self.weights[:, tiles]
        return weights

    def get_one_hot_state(self, state, action=None):
        tiles = self.tile_coder.get_activated_tiles(state)
        one_hot_state = np.zeros(self.iht_size)
        one_hot_state[tiles] = 1
        return one_hot_state

    # == Weights computing functions ==================================================================================

    def compute_weights(self, learning_rate, delta, state, action):
        if self.type == "tile coder":
            grad = np.ones(self.num_tilings)
            tiles = self.tile_coder.get_activated_tiles(state)
            self.weights[action, tiles] += (learning_rate / self.num_tilings) * delta * grad

    def compute_weights_with_eligibility_traces(self, learning_rate, delta, eligibility_traces):
        if self.type == "tile coder":
            self.weights += (learning_rate / self.num_tilings) * delta * eligibility_traces
        else:
            print(f'eligibility traces not yet implemented for {self.type}')

    def compute_weights_with_dutch_traces(self, learning_rate, delta, state, action, eligibility_traces, action_value,
                                          old_action_value):
        if self.type == "tile coder":
            self.weights += (learning_rate / self.num_tilings) * (delta + action_value - old_action_value) * eligibility_traces
            tiles = self.tile_coder.get_activated_tiles(state)
            self.weights[action, tiles] -= (learning_rate / self.num_tilings) * (action_value - old_action_value)
        else:
            print(f'dutch traces not yet implemented for {self.type}')

    def compute_weights_with_reinforce(self, learning_rate, discount_factor, G, pw, state, action):
        if self.type == "tile coder":
            tiles = self.tile_coder.get_activated_tiles(state)
            self.weights[:, tiles] = learning_rate * discount_factor ** pw * G