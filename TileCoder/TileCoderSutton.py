import numpy as np
from TileCoder.IHT import *

class TileCoderSutton:
    def __init__(self, params):
        self.max_size = None  # maximum number of distinct points the agent can go to in the parameter space
        self.iht = None
        self.num_tilings = None
        self.num_tiles = None
        self.min_values = None
        self.max_values = None

        self.set_params_from_dict(params)

        self.set_other_params()

    def set_params_from_dict(self, params):
        self.max_size = params["max_size"]
        self.num_tilings = params["num_tilings"]
        self.num_tiles = params["num_tiles"]
        self.min_values = np.array(params["min_values"])
        self.max_values = np.array(params["max_values"])

    def set_other_params(self):
        self.iht = IHT(self.max_size)

    def get_activated_tiles(self, values):
        """ It is important to remember that the scaling has for only purpose to make the range of the values equal to
        the number of tiles along the dimension.
        """
        # rescaling the values so they are in the interval [0; num_tiles]
        scaled_values = list(((np.array(values) - self.min_values) * self.num_tiles) / (self.max_values - self.min_values))
        tiles_activated = IHT.tiles(self.iht, self.num_tilings, scaled_values)
        return tiles_activated


if __name__ == "__main__":
    tile_coder = TileCoderSutton({
        'max_size': 2048,
        'num_tilings': 8,
        'num_tiles': 8,
        'min_values': [-0.6, -0.07],
        'max_values': [1, 0.07]
    })
    tiles_values = tile_coder.get_activated_tiles([0,0])
    print(tiles_values)

