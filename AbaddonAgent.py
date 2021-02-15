import numpy as np

class AbaddonAgent:
    def __init__(self, params={}):
        self.seed = None

        self.set_params_from_dict(params)

    # ====== Initialization functions =======================================================

    def set_params_from_dict(self, params={}):
        self.init_seed(params.get("seed", None))

    def init_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    def set_seed(self, seed):
        if seed:
            self.seed = seed
            set_random_seed(self.seed)

    # ====== Action choice related functions =================================================
    
    def choose_greedy_act(self, state):
        regions = state['regions']
        ang_scores = np.zeros(360)
        for i in range(int(len(regions) / 3)):
            reg_dist = regions[3 * i]
            reg_ang = regions[3 * i + 1]
            reg_detec = regions[3 * i + 2]
            if not((reg_dist == 0) and (reg_ang == 0) and (reg_detec == 0)):
                if reg_detec == 0:
                    ang_scores[round(reg_ang)] +=1
        action_ang = ang_scores.argmax()
        if action_ang > 180:
            action_ang = action_ang - 360
        return action_ang

    # ====== Agent core functions ============================================================

    def start(self, state):
        # getting actions
        current_action = self.choose_greedy_act(state)

        return current_action


    def step(self, state, reward):
        current_action = self.choose_greedy_act(state)

        return current_action

    def end(self, state, reward):
        pass
