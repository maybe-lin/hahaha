from snake_game import SnakeGameAI, DQNAgent
import numpy as np

if __name__ == "__main__":

    env = SnakeGameAI()
    state_size = 4  # 状态的大小
    action_size = 3  # 动作的数量：前进、左转、右转
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32


    for e in range(episodes):
        if e % 100 == 0:  # 每100回合渲染一次
            env.render()

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e + 1}/{episodes}, Score: {env.score}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)




