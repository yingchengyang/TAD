# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Planar Walker Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2
_PROSTRATE_HEIGHT_LOW = 0.0
_PROSTRATE_HEIGHT_HIGH = 0.2
_PROSTRATE_HEIGHT_MARGIN = 0.1

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8
_SPIN_SPEED = 5

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('walker.xml'), common.ASSETS


@SUITE.add('benchmarking')
def prostrate(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=0, random=random, height_low=_PROSTRATE_HEIGHT_LOW,
                        height_high=_PROSTRATE_HEIGHT_HIGH, height_margin=_PROSTRATE_HEIGHT_MARGIN)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED,
                        flip=True,
                        random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=0, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def walk_speed(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None,
               target_speed=_WALK_SPEED, speed_bound=0.0):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_WALK_SPEED, random=random,
                        target_speed=target_speed, speed_bound=speed_bound)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos['torso', 'z']

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ['xx', 'xz']].ravel()

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class PlanarWalker(base.Task):
    """A planar walker task."""

    def __init__(self, move_speed=0.0, random=None,
                 height_low=_STAND_HEIGHT, height_high=float('inf'),
                 height_margin=_STAND_HEIGHT/2, target_speed=_WALK_SPEED,
                 flip=False,
                 speed_bound=0.0):
        """Initializes an instance of `PlanarWalker`.
        Args:
          move_speed: A float. If this value is zero, reward is given simply for
            standing up. Otherwise this specifies a target horizontal velocity for
            the walking task.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._move_speed = move_speed
        self._height_low = height_low
        self._height_high = height_high
        self._height_margin = height_margin
        self._walk_speed = target_speed
        self._speed_bound = speed_bound
        self._flip = flip
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        In 'standing' mode, use initial orientation and small velocities.
        In 'random' mode, randomize joint angles and let fall to the floor.
        Args:
          physics: An instance of `Physics`.
        """
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     # bounds=(self._height, float('inf')),
                                     bounds=(self._height_low, self._height_high),
                                     margin=self._height_margin)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._flip:
            move_reward = rewards.tolerance(physics.angmomentum(),
                                            bounds=(_SPIN_SPEED, float('inf')),
                                            margin=_SPIN_SPEED,
                                            value_at_margin=0,
                                            sigmoid='linear')
            return stand_reward * (5 * move_reward + 1) / 6
        elif self._move_speed == 0:
            return stand_reward
        elif self._speed_bound == 0.0:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(self._move_speed, float('inf')),
                                            margin=self._move_speed / 2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5 * move_reward + 1) / 6
        else:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(self._walk_speed-self._speed_bound,
                                                    self._walk_speed+self._speed_bound),
                                            margin=self._speed_bound / 2,
                                            value_at_margin=0.0,
                                            sigmoid='linear')
            # return stand_reward * (5 * move_reward + 1) / 6
            return stand_reward * move_reward
