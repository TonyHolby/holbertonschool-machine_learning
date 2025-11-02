#!/usr/bin/env python3
"""
    A python script train.py that utilizes keras, keras-rl2, and gymnasium
    to train an agent that can play Atari's Breakout
"""
import numpy as np
from PIL import Image
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
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
        obs, info = self.env.reset(**kwargs)
        frame = preprocess_observation(obs)
        self.frames = np.stack([frame]*self.k, axis=-1)

        return self.frames

    def step(self, action):
        """
            Resets the environment and initialize the frame stack.

            Args:
                **kwargs (dict): an additional keyword arguments
                    passed to "env.reset()".

            Returns:
                The initial stacked frames of shape (84, 84, k).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        frame = preprocess_observation(obs)
        self.frames = np.roll(self.frames, shift=-1, axis=-1)
        self.frames[:, :, -1] = frame

        return self.frames, reward, done, info


env = gym.make("ALE/Breakout-v5", render_mode=None)
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
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.0,
                              value_min=0.1,
                              value_test=0.05,
                              nb_steps=500000)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=50000,
               target_model_update=10000,
               policy=policy,
               gamma=0.99,
               train_interval=4,
               delta_clip=1.0)

dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

nb_steps = 2000000
dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=2)

dqn.save_weights("policy.h5", overwrite=True)
env.close()
