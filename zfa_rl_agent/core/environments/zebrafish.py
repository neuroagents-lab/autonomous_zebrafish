"""Procedurally generated Swimmer domain."""
import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np

from dm_control.suite.wrappers import pixels
from shimmy import DmControlCompatibilityV0
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from zfa_rl_agent.core.utils.logger import logger
from zfa_rl_agent.core.agent.extractors import VecImageStack

_DEFAULT_TIME_LIMIT = 30 # this corresponds to 1000 timesteps
_CONTROL_TIMESTEP = .03  # (Seconds)

SUITE = containers.TaggedTasks()

def swimmer_venv(parallel, n_envs, seed, monitor_dir, force_magnitude, view_render_args=dict(), obs_render_args=dict()):
    obs_render_args = obs_render_args or {'height': 64, 'width': 64, 'camera_id': 2}
    view_render_args = view_render_args or {'height': 256, 'width': 256, 'camera_id': 0}

    if parallel:
      logger.info("Vectorizing with SubprocVecEnv")
      vec_env = make_vec_env(lambda: swimmer(force_magnitude=force_magnitude, obs_render_args=obs_render_args, view_render_args=view_render_args), 
                              n_envs=n_envs, seed=seed, monitor_dir=monitor_dir, vec_env_cls=SubprocVecEnv)
    else:
      vec_env = make_vec_env(lambda: swimmer(force_magnitude=force_magnitude, obs_render_args=obs_render_args, view_render_args=view_render_args),
                                n_envs=n_envs, seed=seed, monitor_dir=monitor_dir, vec_env_cls=DummyVecEnv)

    env = VecTransposeImage(vec_env)
   # env = VecImageStack(env, n_stack=2, image_key='pixels', channels_order='first')
    return env

def swimmer(n_links=6,
                     force_magnitude=0.0,
                     obs_render_args={'height': 64, 'width': 64, 'camera_id': 2},
                     view_render_args={'height': 256, 'width': 256, 'camera_id': 0},
                     time_limit=_DEFAULT_TIME_LIMIT, 
                     random=None):
  
  """Returns a swimmer with n links."""
  dm_env = _make_swimmer_dm(n_links, force_magnitude, obs_render_args, time_limit, random=random)
  env = DmControlCompatibilityV0(dm_env, render_mode='rgb_array', render_kwargs=view_render_args)
  return env

def swimmer_dm(n_links=6,
                     force_magnitude=0.0,
                     render_args = {'height': 64, 'width': 64, 'camera_id': 2},
                     time_limit=_DEFAULT_TIME_LIMIT, 
                     random=None):
  
  """Returns a swimmer with n links."""
  return _make_swimmer_dm(n_links, force_magnitude, render_args, time_limit, random=random)

def _make_swimmer_dm(n_joints,
                    force_magnitude=1.0,
                    render_args={'height': 64, 'width': 64, 'camera_id': 2},
                    time_limit=_DEFAULT_TIME_LIMIT, 
                    random=None, 
                    environment_kwargs=None,
                    ):

  """Returns a swimmer control environment."""
  model_string, assets = get_model_and_assets(n_joints)
  physics = Physics.from_xml_string(model_string, assets=assets)
  physics.force_magnitude = force_magnitude  # Set the force magnitude
  task = drift(random=random)
  
  environment_kwargs = environment_kwargs or {}
  env = control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=_CONTROL_TIMESTEP,
    **environment_kwargs,
    )
  env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_args)
  return env

def apply_fluid_drift(physics, force_magnitude):
    """
    Apply a constant backward force to simulate fluid drift.

    Args:
        physics: The `mjcf.Physics` object or `mjData` instance.
        force_magnitude: Magnitude of the applied force.
    """
    nose_to_target_local = physics.nose_to_target()  # shape (2,)

    force_local = -nose_to_target_local
    force_local  = np.append(force_local, 0.0)
    head_orientation = physics.named.data.xmat['head'].reshape(3, 3)
    force_global = head_orientation.dot(force_local)
    force_global = force_magnitude * force_global / np.linalg.norm(force_global)
    #force_global = force_magnitude * np.array([-1.0, 0.0, 0.0])
    n_bodies = physics.model.nbody
    force_matrix = np.tile(force_global, (n_bodies, 1))

    # Apply the cartesian force (fx, fy, fz). 
    physics.data.xfrc_applied[:, :3] = force_matrix 



class Physics(mujoco.Physics):
  """Physics simulation with additional features for the zebrafish domain."""
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_magnitude = 1.0  # Adjust this for stronger or weaker drift forces

  def step(self, nstep: int = 1):
      """Steps the simulation and applies the fluid drift force."""
      apply_fluid_drift(self, self.force_magnitude)
      super().step(nstep)

  def nose_to_target(self):
    """Returns a vector from nose to target in local coordinate of the head."""
    nose_to_target = (self.named.data.geom_xpos['target'] -
                      self.named.data.geom_xpos['nose'])
    head_orientation = self.named.data.xmat['head'].reshape(3, 3)
    return nose_to_target.dot(head_orientation)[:2]

  def nose_to_target_dist(self):
    """Returns the distance from the nose to the target."""
    return np.linalg.norm(self.nose_to_target())

  def body_velocities(self):
    """Returns local body velocities: x,y linear, z rotational."""
    xvel_local = self.data.sensordata[12:].reshape((-1, 6))
    vx_vy_wz = [0, 1, 5]  # Indices for linear x,y vels and rotational z vel.
    return xvel_local[:, vx_vy_wz].ravel()

  def joints(self):
    """Returns all internal joint angles (excluding root joints)."""
    return self.data.qpos[3:].copy()


class drift(base.Task):
  """A swimmer `Task` to reach the target or just swim."""

  def __init__(self, random=None):
    """Initializes an instance of `Swimmer`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Initializes the swimmer orientation to [-pi, pi) and the relative joint
    angle of each joint uniformly within its range.

    Args:
      physics: An instance of `Physics`.
    """
    # Random joint angles:
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)

    # Random target position.
    close_target = self.random.rand() < .2  # Probability of a close target.
    target_box = .3 if close_target else 2
    xpos, ypos = self.random.uniform(-target_box, target_box, size=2)
    physics.named.model.geom_pos['target', 'x'] = xpos
    physics.named.model.geom_pos['target', 'y'] = ypos
    physics.named.model.light_pos['target_light', 'x'] = xpos
    physics.named.model.light_pos['target_light', 'y'] = ypos

    # # Hide target by setting alpha to 0.
    # physics.named.model.mat_rgba['target', 'a'] = 0
    # physics.named.model.mat_rgba['target_default', 'a'] = 0
    # physics.named.model.mat_rgba['target_highlight', 'a'] = 0

    # # Make the target non-collidable.
    # physics.named.model.geom_contype['target'] = 0
    # physics.named.model.geom_conaffinity['target'] = 0

    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of joint angles, body velocities and target."""
    obs = collections.OrderedDict()
    concat_obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    obs['to_target'] = physics.nose_to_target()
    obs['body_velocities'] = physics.body_velocities()

    concat_obs['proprio'] = np.concatenate([obs['joints'], obs['body_velocities'], obs['to_target']])
    return concat_obs

  def get_reward(self, physics):
    """Returns a smooth reward."""
    target_size = physics.named.model.geom_size['target', 0]
    return rewards.tolerance(physics.nose_to_target_dist(),
                             bounds=(0, target_size),
                             margin=5*target_size,
                             sigmoid='long_tail')

assets_dir = "/home/rdkeller/zebrafish_agent/zfa_rl_agent/core/environments/"
ASSETS = {}
for key, value in common.ASSETS.items():
    new_key = assets_dir + key
    ASSETS[new_key] = value

def get_model_and_assets(n_joints):
  """Returns a tuple containing the model XML string and a dict of assets.

  Args:
    n_joints: An integer specifying the number of joints in the swimmer.

  Returns:
    A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
    `{filename: contents_string}` pairs.
  """
  return _make_model(n_joints), ASSETS

def _make_body(body_index):
  """Generates an xml string defining a single physical body.
  
  Arguments:
  body_index: index of the body
  
  Returns:
  XML string defining the body
  """
  body_name = 'segment_{}'.format(body_index)
  visual_name = 'visual_{}'.format(body_index)
  inertial_name = 'inertial_{}'.format(body_index)
  body = etree.Element('body', name=body_name)
  body.set('pos', '0 .1 0')
  etree.SubElement(body, 'geom', {'class': 'visual', 'name': visual_name})
  etree.SubElement(body, 'geom', {'class': 'inertial', 'name': inertial_name})
  return body

def _make_model(n_bodies):
  """Generates an xml string defining a swimmer with `n_bodies` bodies."""
  if n_bodies < 3:
    raise ValueError('At least 3 bodies required. Received {}'.format(n_bodies))
  mjcf = etree.fromstring(common.read_model('swimmer.xml'))
  head_body = mjcf.find('./worldbody/body')
  actuator = etree.SubElement(mjcf, 'actuator')
  sensor = etree.SubElement(mjcf, 'sensor')

  # Change ground background 
  asset = etree.SubElement(mjcf, 'asset')
  etree.SubElement(asset, 'texture', {
    'name': 'grass_tex',
    'type': '2d',
    'file': '/home/rdkeller/zebrafish_agent/zfa_rl_agent/core/environments/common/OutdoorGrassFloorD.png'
    #'file': '/Users/reecekeller/Documents/neuroagents/zebrafish_agent/zfa_rl_agent/core/environments/common/OutdoorGrassFloorD.png'

  })
  etree.SubElement(asset, 'material', {
    'name': 'grass_mat',
    'texture': 'grass_tex',
    'specular': '0.3',
    'shininess': '0.1'
  })

  ground_geom = mjcf.find('.//geom[@name="ground"]')
  if ground_geom is not None:
    ground_geom.set('material', 'grass_mat')
    ground_geom.set('size', '5 5 0.1')

  parent = head_body
  for body_index in range(n_bodies - 1):
    site_name = 'site_{}'.format(body_index)
    child = _make_body(body_index=body_index)
    child.append(etree.Element('site', name=site_name))
    joint_name = 'joint_{}'.format(body_index)
    joint_limit = 360.0/n_bodies
    joint_range = '{} {}'.format(-joint_limit, joint_limit)
    child.append(etree.Element('joint', {'name': joint_name,
                                         'range': joint_range}))
    motor_name = 'motor_{}'.format(body_index)
    actuator.append(etree.Element('motor', name=motor_name, joint=joint_name))
    velocimeter_name = 'velocimeter_{}'.format(body_index)
    sensor.append(etree.Element('velocimeter', name=velocimeter_name,
                                site=site_name))
    gyro_name = 'gyro_{}'.format(body_index)
    sensor.append(etree.Element('gyro', name=gyro_name, site=site_name))
    parent.append(child)
    parent = child

  # Move tracking cameras further away from the swimmer according to its length.
  cameras = mjcf.findall('./worldbody/body/camera')
  scale = n_bodies / 6.0
  for cam in cameras:
    if cam.get('mode') == 'trackcom':
      old_pos = cam.get('pos').split(' ')
      new_pos = ' '.join([str(float(dim) * scale) for dim in old_pos])
      cam.set('pos', new_pos)

  return etree.tostring(mjcf, pretty_print=True)


