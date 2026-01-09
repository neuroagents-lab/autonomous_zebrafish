"""Swimmer environment with moving visual gratings."""
import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from lxml import etree
import numpy as np
from PIL import Image

from dm_control.suite.wrappers import pixels
from shimmy import DmControlCompatibilityV0
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from zfa_rl_agent.core.multi_processing import DistVecEnv
from zfa_rl_agent.core.utils.logger import logger
from zfa_rl_agent.core.agent.extractors import VecImageStack

_DEFAULT_TIME_LIMIT = 30  # this corresponds to 1000 timesteps
_CONTROL_TIMESTEP = .03   # (Seconds)

ZEBRAFISH_LENGTH = 0.0045  # Zebrafish length in meters (4.5 mm)
GRATING_WIDTH = 0.0025  # Width of each stripe in meters (2.5 mm)
GRATING_SPACING = 0.0025  # Spacing between stripes in meters (2.5 mm)
SCALE_RATIO = ZEBRAFISH_LENGTH / (0.08 + 0.1 * 6)  # Ratio of zebrafish length to swimmer length

assets_dir = os.path.dirname(os.path.abspath(__file__))
striped_asset_path = os.path.join(assets_dir, "common/striped_texture.png")

SUITE = containers.TaggedTasks()


## Environment creation functions

def swimmer_grating_venv(parallel, n_envs, seed, monitor_dir, 
                         grating_speed, view_render_args=dict(), obs_render_args=dict()):
    obs_render_args = obs_render_args or {'height': 64, 'width': 64, 'camera_id': 2}
    view_render_args = view_render_args or {'height': 256, 'width': 256, 'camera_id': 0}

    #view_render_args = obs_render_args.copy()
    if parallel:
      logger.info("Vectorizing with SubprocVecEnv")
      vec_env = make_vec_env(lambda: swimmer_grating(grating_speed=grating_speed, obs_render_args=obs_render_args, view_render_args=view_render_args), 
                              n_envs=n_envs, seed=seed, monitor_dir=monitor_dir, vec_env_cls=SubprocVecEnv)
    else:
      logger.info("Vectorizing with DummyVecEnv")
      vec_env = make_vec_env(lambda: swimmer_grating(grating_speed=grating_speed, obs_render_args=obs_render_args, view_render_args=view_render_args), 
                             n_envs=n_envs, seed=seed, monitor_dir=monitor_dir, vec_env_cls=DummyVecEnv)
    
    env = VecTransposeImage(vec_env)
    #env = VecImageStack(env, n_stack=2, image_key='pixels', channels_order='first')
    return env

def swimmer_grating(n_links=6,
                   grating_speed=0.01,
                   obs_render_args={'height': 64, 'width': 64, 'camera_id': 2},
                   view_render_args={'height': 256, 'width': 256, 'camera_id': 0},
                   time_limit=_DEFAULT_TIME_LIMIT, 
                   random=None):
  """Returns a swimmer with n links and moving grating."""
  dm_env = _make_swimmer_grating_dm(n_links, grating_speed, obs_render_args, time_limit, random=random)
  env = DmControlCompatibilityV0(dm_env, render_mode='rgb_array', render_kwargs=view_render_args)
  return env

def swimmer_grating_dm(n_links=6,
                      grating_speed=0.01,
                      render_args={'height': 64, 'width': 64, 'camera_id': 2},
                      time_limit=_DEFAULT_TIME_LIMIT, 
                      random=None):
  
  """Returns a swimmer with n links and moving grating for dm_control."""
  return _make_swimmer_grating_dm(n_links, grating_speed, render_args, time_limit, random=random)


## Base task creation function

def _make_swimmer_grating_dm(n_joints=6,
                            grating_speed=0.01,
                            render_args={'height': 64, 'width': 64, 'camera_id': 2}, 
                            time_limit=_DEFAULT_TIME_LIMIT, 
                            random=None, 
                            environment_kwargs=None):

  """Returns a swimmer control environment with moving gratings for dm_control."""
  model_string, assets = get_model_and_assets(n_joints)
  physics = Physics.from_xml_string(model_string, assets=assets)
  physics.grating_speed = grating_speed  # Set the grating speed
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


## Environment physics

def move_visual_grating(physics, grating_speed, grating_period=None):
    """
    Move the visual grating by updating the joint position of the floor.
    
    Args:
        physics: The `mjcf.Physics` object or `mjData` instance.
        grating_speed: Speed of the grating movement.
    """
    # Find the joint for the moving floor
    try:
        # center floor around head x position
        head_pos = physics.named.data.geom_xpos["head"]
        x_joint_index = physics.model.name2id("floor_joint_x", "joint")
        physics.data.qpos[x_joint_index] = head_pos[0]

        # move grating to maintain phase around head y position
        y_joint_index = physics.model.name2id("floor_joint_y", "joint")
        curr_joint_pos = physics.data.qpos[y_joint_index]
        new_joint_pos = curr_joint_pos - grating_speed
        if grating_period is not None:
            head_dist = head_pos[1] - new_joint_pos
            head_dist = round(head_dist / grating_period) * grating_period
            new_joint_pos = new_joint_pos + head_dist
        physics.data.qpos[y_joint_index] = new_joint_pos
    except KeyError:
        # If joint not found, we'll assume it doesn't exist and skip
        return

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the swimmer grating domain."""
  def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grating_speed = 0.01  # Default grating speed
        self.head_fixed = True

        # calculate grating period
        scaled_ground_size = 4.0 * SCALE_RATIO  # Based on ground size in MuJoCo
        num_pixels = int(scaled_ground_size * 10000)  # High resolution
        num_pixels = max(num_pixels, 512)  # Minimum resolution
        pixel_size = scaled_ground_size / num_pixels
        line_width_px = max(1, int(GRATING_WIDTH / pixel_size))
        line_spacing_px = max(1, int(GRATING_SPACING / pixel_size))
        grating_period = 4.0 * int(line_width_px + line_spacing_px) / num_pixels
        self.grating_period = grating_period

  def step(self, nstep: int = 1):
      """Steps the simulation and moves the visual grating."""
      if self.head_fixed:  # get initial head position
          x_init = self.data.qpos[self.model.name2id("rootx", "joint")]
          y_init = self.data.qpos[self.model.name2id("rooty", "joint")]
          z_init = self.data.qpos[self.model.name2id("rootz", "joint")]

      move_visual_grating(self, self.grating_speed, self.grating_period)
      super().step(nstep)

      if self.head_fixed:  # reset head position
          self.data.qpos[self.model.name2id("rootx", "joint")] = x_init
          self.data.qpos[self.model.name2id("rooty", "joint")] = y_init
          self.data.qpos[self.model.name2id("rootz", "joint")] = z_init

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
    return self.data.qpos[3:8].copy()


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
    # Fixed initial joint angles:
    for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, 'joint')
        if "joint_" in joint_name or "root" in joint_name:
            physics.named.data.qpos[joint_name] = 0

    # Fixed target position.
    physics.named.model.geom_pos['target', 'x'] = 0
    physics.named.model.geom_pos['target', 'y'] = -1
    physics.named.model.light_pos['target_light', 'x'] = 0
    physics.named.model.light_pos['target_light', 'y'] = -1

    # Make target and grid invisible
    physics.named.model.mat_rgba['grid', 'a'] = 0
    physics.named.model.mat_rgba['target', 'a'] = 0
    physics.named.model.mat_rgba['target_default', 'a'] = 0
    physics.named.model.mat_rgba['target_highlight', 'a'] = 0

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
  

## Model and assets

def generate_striped_texture(image_size, line_width, line_spacing):
    """
    Generate a black-and-blue striped texture with adjustable parameters.
    
    Args:
        image_size: Tuple of (width, height) for the image
        line_width: Width of each stripe in pixels
        line_spacing: Spacing between stripes in pixels
    
    Returns:
        numpy.ndarray: The generated texture as a RGB array
    """
    width, height = image_size
    
    # Create a black image (all channels initialized to 0)
    texture = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw horizontal blue stripes
    for y in range(0, height, int(line_width + line_spacing)):
        y_end = min(y + line_width, height)  # avoid out-of-bounds
        texture[y:y_end, :, 0] = 0    # Red channel
        texture[y:y_end, :, 1] = 0    # Green channel
        texture[y:y_end, :, 2] = 255  # Blue channel

    return texture

def create_grating_texture(save_path=striped_asset_path):
    """Create and save the striped grating texture."""    
    # Calculate appropriate scale for texture
    scaled_ground_size = 4.0 * SCALE_RATIO  # Based on ground size in MuJoCo
    num_pixels = int(scaled_ground_size * 10000)  # High resolution
    num_pixels = max(num_pixels, 512)  # Minimum resolution
    
    pixel_size = scaled_ground_size / num_pixels
    
    # Convert mm measurements to pixels
    line_width_px = max(1, int(GRATING_WIDTH / pixel_size))
    line_spacing_px = max(1, int(GRATING_SPACING / pixel_size))
    
    # Generate texture
    image_size = (num_pixels, num_pixels)
    striped_texture = generate_striped_texture(image_size, line_width_px, line_spacing_px)
    image = Image.fromarray(striped_texture)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        
    return striped_texture

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
  """Generates an xml string defining a swimmer with `n_bodies` bodies and a moving floor."""
  if n_bodies < 3:
    raise ValueError('At least 3 bodies required. Received {}'.format(n_bodies))
  mjcf = etree.fromstring(common.read_model('swimmer.xml'))
  head_body = mjcf.find('./worldbody/body')
  actuator = etree.SubElement(mjcf, 'actuator')
  sensor = etree.SubElement(mjcf, 'sensor')

  # Add striped texture and material
  asset = mjcf.find('./asset')
  if asset is None:
    asset = etree.SubElement(mjcf, 'asset')
    
  # Add texture for striped pattern
  etree.SubElement(asset, 'texture', {
    'name': 'striped_tex',
    'type': '2d',
    'file': striped_asset_path,
  })
  
  # Add material using the texture
  etree.SubElement(asset, 'material', {
    'name': 'striped_mat',
    'texture': 'striped_tex',
    'specular': '0.3',
    'shininess': '0.1'
  })

  # Create a moving floor with our texture
  worldbody = mjcf.find('./worldbody')
  moving_floor = etree.SubElement(worldbody, 'body', name='moving_floor_body')
  etree.SubElement(moving_floor, 'geom', {
    'name': 'moving_floor',
    'type': 'box',
    'size': '2 2 0.0001',
    'material': 'striped_mat',
    'conaffinity': '0',
    'contype': '0'
  })
  
  # Add a slider joint to move the floor
  etree.SubElement(moving_floor, 'joint', {
    'name': 'floor_joint_x',
    'type': 'slide',
    'axis': '1 0 0',
    'limited': 'false'
  })
  etree.SubElement(moving_floor, 'joint', {
    'name': 'floor_joint_y',
    'type': 'slide',
    'axis': '0 1 0',
    'limited': 'false'
  })

  # Build swimmer body parts
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