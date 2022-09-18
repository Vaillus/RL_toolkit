import numpy as np


class TileCoder:
    """
    My attempt at doing a tile coder.
    """
    def __init__(self, params):
        self.num_tilings = None
        self.num_tiles = None
        self.min_values = None
        self.max_values = None

        self.values_ranges = None
        self.tile_size = None
        self.tilings_absolute_origin = None
        self.tiling_size = None
        self.tilings_delta = None
        self.tilings_origins = None
        self.is_action_in_dims = None

        self.set_params_from_dict(params)
        self.check_initialization_input()
        self.set_other_params()

    def set_params_from_dict(self, params):
        """

        :param params:
        :return:
        """
        self.num_tilings = params["num_tilings"]
        self.num_tiles = np.array(params["num_tiles"])
        self.min_values = np.array(params["min_values"])
        self.max_values = np.array(params["max_values"])
        self.is_action_in_dims = params["is_action_in_dims"]

    def check_initialization_input(self):
        assert self.num_tiles.shape == self.min_values.shape == self.max_values.shape, "Dimensions of num_tiles, min_values and max_values must be the same."
        assert np.all(
            self.min_values < self.max_values), "All max values must be strictly greater than their corresponding min value"
        assert self.num_tilings > 0, "There must be at least 1 tiling"
        assert np.all(self.num_tiles > 1), "There must be at least 2 tiles per dimension"

    def set_other_params(self):
        self.num_dim = self.min_values.shape[0]
        self.values_ranges = self.max_values - self.min_values
        self.tile_size = self.values_ranges / (self.num_tiles - 1)
        self.tilings_absolute_origin = self.min_values - self.tile_size
        self.tiling_size = np.prod(self.num_tiles)
        self.tilings_delta = self.tile_size / self.num_tilings
        self.set_tilings_origins()

    def set_tilings_origins(self):
        self.tilings_origins = np.zeros([self.num_tilings, self.min_values.shape[0]])
        for n_tiling in range(self.num_tilings):
            self.tilings_origins[n_tiling] = self.tilings_absolute_origin + (n_tiling + 0.5) * self.tilings_delta
        for n_dim in range(self.num_dim):
            np.random.shuffle(self.tilings_origins[:, n_dim])
        self.tilings_origins[:, -1] = self.min_values[-1]

    def get_tilings_values(self, values):

        values = self.check_format_input_values(values)

        tile_activated = np.zeros([self.num_tilings, self.num_dim])
        for n_tiling in range(self.num_tilings):
            for n_dim in range(self.num_dim):
                for n_tile in range(self.num_tiles[n_dim]):
                    if values[n_dim] >= self.tilings_origins[n_tiling, n_dim] + n_tile * self.tile_size[n_dim]:
                        tile_activated[n_tiling, n_dim] = n_tile

        tiling_values = np.zeros(self.num_tilings)
        for n_tiling in range(self.num_tilings):
            for n_dim in range(self.num_dim):
                tiling_values[n_tiling] += tile_activated[n_tiling, n_dim] * np.prod(self.num_tiles[:n_dim])

        tiling_values = tiling_values.astype(int)

        return tiling_values

    def check_format_input_values(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        assert values.shape == self.num_tiles.shape, \
            "The dimension of your input vector doesn't match the tile coder dimension"
        assert np.all(self.min_values <= values) and np.all(values <= self.max_values), \
            "The input values are not between the tilecoders min and max values"

        return values

    @staticmethod
    def add_action_to_state(params, action_info):
        params['min_values'] = np.append(params['min_values'], action_info['min_action'])
        params['max_values'] = np.append(params['max_values'], action_info['max_action'])
        params['num_tiles'] = np.append(params['num_tiles'], action_info['num_actions'])

        return params

    def get_best_action(self, state, weights):
        action_values = np.array([])
        # TODO : change num_tiles[-1], find a way to explicitly use the number of actions
        for action in range(self.num_tiles[-1]):
            tiles_values = self.get_tilings_values(np.append(state, action))
            #print(tiles_values)
            action_value = np.sum(weights[tiles_values])
            action_values = np.append(action_values, action_value)
        return np.argmax(action_values)

    def choose_epsilon_greedy_action(self, state, weights, epsilon):
        best_action = self.get_best_action(state, weights)
        if np.random.uniform() < epsilon:
            return best_action
        else:
            # TODO : change num_tiles[-1], find a way to explicitly use the number of actions
            return np.random.randint(self.num_tiles[-1])



if __name__ == "__main__":
    import gym

    env = gym.make("MountainCar-v0")

    params = {
        'num_tilings': 4,
        'num_tiles': [8, 8],
        'min_values': env.observation_space.low,
        'max_values': env.observation_space.high,
        'is_action_in_dims': True
    }

    action_info = {
        'min_action': 0,
        'max_action': 2,
        'num_actions': 3
    }

    params = TileCoder.add_action_to_state(params, action_info)

    tc = TileCoder(params)

    weights = np.zeros(shape=np.prod(tc.num_tiles))
    step_size = 0.5 / params['num_tilings']
    gamma = 0.5
    epsilon = 0.95

    EPISODES = 2000
    SHOW_EVERY = 100
    success_count = 0
    print(tc.tilings_origins)
    #print(tc.get_tilings_values(np.array([0, 0.1, 1])))
    for episode in range(EPISODES):
        done = False

        state = env.reset()
        action = tc.get_best_action(state, weights)
        # action = tc.choose_epsilon_greedy_action(state, weights, epsilon)
        tiles_values = tc.get_tilings_values(np.append(state, action))
        #print(tiles_values)
        #print(tc.tilings_origins)

        while not done:

            new_state, reward, done, _ = env.step(action)

            if episode % SHOW_EVERY == 0:
                    env.render()

            action_value = np.sum(weights[tiles_values])
            if done:
                weights[tiles_values] += step_size * (reward - action_value)
                if new_state[0] >= env.goal_position:
                    #print("succeeeed!!!")
                    success_count += 1
            else:

                new_action = tc.choose_epsilon_greedy_action(new_state, weights, epsilon)

                new_tiles_values = tc.get_tilings_values(np.append(new_state, new_action))

                new_action_value = np.sum(weights[new_tiles_values])
                weights[tiles_values] += step_size * (reward + gamma * new_action_value - action_value)
                state = new_state
                action = new_action
                action_value = np.sum(weights[tiles_values])
                tiles_values = new_tiles_values

        if episode % SHOW_EVERY == 0:
            print(f'EPISODE {episode}:')
            print(f'    pourcentage of success: {success_count/SHOW_EVERY * 100}%')
            success_count = 0
            #print(weights[:110])

    env.close()
