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

    def compute_weights(self, learning_rate, delta, state, action):
        if self.type == "tile coder":
            grad = np.ones(self.num_tilings)
            tiles = self.tile_coder.get_activated_tiles(state)
            self.weights[action, tiles] += (learning_rate / self.num_tilings) * delta * grad
