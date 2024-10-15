import numpy as np
import random
import pygame
from pygame.locals import *
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 游戏常量
WIDTH, HEIGHT = 500, 500
BLOCK_SIZE = 20
FPS = 5


# 贪吃蛇
class SnakeGameAI:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('AI Snake')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = (1, 0)
        self.food = self._place_food()
        self.score = 0
        return self._get_state()

    def _place_food(self):
        while True:
            food = (random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                    random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE)
            if food not in self.snake:
                return food

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        state = [
            head_x - food_x, head_y - food_y,
            self.direction[0], self.direction[1],
        ]
        return np.array(state, dtype=int)

    def step(self, action):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        if action == 0:  # 向前
            pass
        elif action == 1:  # 左转
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2:  # 右转
            self.direction = (self.direction[1], -self.direction[0])

        new_head = (self.snake[0][0] + self.direction[0] * BLOCK_SIZE,
                    self.snake[0][1] + self.direction[1] * BLOCK_SIZE)

        self.snake = [new_head] + self.snake[:-1]

        if new_head == self.food:
            self.snake.append(self.snake[-1])
            self.food = self._place_food()
            self.score += 1

        reward = 1 if new_head == self.food else 0
        done = self._is_collision(new_head)

        return self._get_state(), reward, done

    def _is_collision(self, pos):
        if (pos[0] < 0 or pos[0] >= WIDTH or pos[1] < 0 or pos[1] >= HEIGHT):
            return True
        if pos in self.snake[1:]:
            return True
        return False

    def render(self):
        self.display.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), (segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255, 0, 0), (self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()
        self.clock.tick(FPS)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(4, input_dim=self.state_size, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

