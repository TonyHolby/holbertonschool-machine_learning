#!/usr/bin/env python3
"""
    A python script play.py that can display a game played by the agent
    trained by train.py.
"""
import numpy as np
from PIL import Image
import gymnasium as gym
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
import tensorflow.keras

if not hasattr(tf.keras, "__version__"):
    tf.keras.__version__ = tf.__version__

tf.compat.v1.disable_eager_execution()


def preprocess_observation(obs):
    """
        Preprocess a raw Atari observation for the DQN agent.
        Converts the input RGB observation to grayscale, resizes it to
        84x84 pixels, and normalizes pixel values to the range [0, 1].

        Args:
            obs (np.ndarray): the raw observation from the Gymnasium
                environment.

        Returns:
            The preprocessed 2D grayscale observation with shape (84, 84)
            and dtype float32.
    """
    img = Image.fromarray(obs)
    img = img.convert("L")
    img = img.resize((84, 84), Image.BILINEAR)

    return np.array(img, dtype="float32") / 255.0


class FrameStackWrapper(gym.Wrapper):
    """ Gymnasium wrapper that maintains a stack of the last k frames. """
    def __init__(self, env, k=4):
        """
            Initializes the FrameStackWrapper.

            Args:
                env (gym.Env): the Gymnasium environment to wrap.
                k (int): number of frames to stack, by default 4.
        """
        super().__init__(env)
        self.k = k
        self.frames = np.zeros((84, 84, k), dtype=np.float32)

    def reset(self, **kwargs):
        """
            Resets the environment and initialize the frame stack.

            Args:
                **kwargs (dict): an additional keyword arguments
                    passed to "env.reset()".

            Returns:
                The initial stacked frames of shape (84, 84, k).
        """
        obs, info = self.env.reset(**kwargs)
        frame = preprocess_observation(obs)
        self.frames = np.stack([frame]*self.k, axis=-1)

        return self.frames

    def step(self, action):
        """
            Takes a step in the environment and update the frame stack.

            Args:
                action (int): the action to take in the environment.

            Returns:
                self.frames: the updated stacked frames of shape (84, 84, k).
                reward: the reward obtained from taking the action.
                done: True if the episode has terminated
                    (either truncated or ended).
                info: an additional information from the environment step.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        frame = preprocess_observation(obs)
        self.frames = np.roll(self.frames, shift=-1, axis=-1)
        self.frames[:, :, -1] = frame

        return self.frames, reward, done, info


env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = FrameStackWrapper(env, k=4)
nb_actions = env.action_space.n

model = Sequential()
model.add(InputLayer(input_shape=(1, 84, 84, 4)))
model.add(Conv2D(16, (8, 8), strides=(4, 4), activation='relu'))
model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

memory = SequentialMemory(limit=100000, window_length=1)
policy = GreedyQPolicy()

dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=0,
    target_model_update=1,
    policy=policy)

dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])
dqn.load_weights("policy.h5")

plt.ion()
fig, ax = plt.subplots(figsize=(10, 12))
image_display = None

nb_episodes = 3
for ep in range(nb_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done and steps < 2000:
        if steps < 10:
            action = 1
        else:
            action = dqn.forward(obs)

        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1

        rgb_frame = env.unwrapped.ale.getScreenRGB()

        if image_display is None:
            image_display = ax.imshow(rgb_frame)
            ax.axis("off")
        else:
            image_display.set_data(rgb_frame)

        lives = env.unwrapped.ale.lives()
        plt.draw()
        plt.pause(0.001)

        if not plt.fignum_exists(fig.number):
            env.close()
            exit(0)

    print(f"Episode {ep+1}: Score={episode_reward:.0f}, Steps={steps}")

plt.ioff()
plt.close()
env.close()
