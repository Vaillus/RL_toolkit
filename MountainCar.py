import numpy as np
import pandas as pd
import gym

class MountainCar:
    def __init__(self):
        print("lol")

    def create_action_values_if_not_exist(self, df_action_values, position, speed):
        if df_action_values.loc[(df_action_values.position == position) & (df_action_values.speed == speed)].shape[0] == 0:
            df_action_values = df_action_values.append(
                {"position": position, "speed": speed, "action": 0, "value": 0}, ignore_index=True)
            df_action_values = df_action_values.append(
                {"position": position, "speed": speed, "action": 1, "value": 0}, ignore_index=True)
            df_action_values = df_action_values.append(
                {"position": position, "speed": speed, "action": 2, "value": 0}, ignore_index=True)
        return df_action_values

    def update_action_value(self, df_action_values, old_position, old_speed, action, reward, new_position, new_speed,
                            alpha=0.2, gamma=0.95):
        df_action_values = create_action_values_if_not_exist(df_action_values, new_position, new_speed)

        old_value = df_action_values.loc[
            (df_action_values.position == old_position) & (df_action_values.speed == old_speed) & (
                        df_action_values.action == action), "value"]
        max_next_value = np.max(df_action_values.loc[(df_action_values.position == new_position) & (
                    df_action_values.speed == new_speed), "value"])
        error = alpha * (reward + gamma * (max_next_value - old_value))

        df_action_values.loc[(df_action_values.position == old_position) & (df_action_values.speed == old_speed) & (
                    df_action_values.action == action), "value"] += error
        return df_action_values

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    env.reset()

    LEARNING_RATE = 0.1
    DISCOUNT = 0.95
    EPISODES = 3000

    SHOW_EVERY = 100

    DISCRETE_OS_SIZE = [20] * env.observation_space.shape[0]
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.int))


    times = []

    for episode in range(EPISODES):
        discrete_state = get_discrete_state(env.reset())
        done = False

        while not done:
            # t1 = time.time()
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)
            if episode % SHOW_EVERY == 0:
                env.render()
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                # Concatenating tuples
                current_q = q_table[discrete_state + (action,)]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = 0
                print("succeeeed!!!")

            discrete_state = new_discrete_state
            # t2 = time.time()
            # times.append(t2-t1)
        # print(f'mean of time loop: {np.mean(times)}')
        if episode % SHOW_EVERY == 0:
            print(episode)

    env.close()
