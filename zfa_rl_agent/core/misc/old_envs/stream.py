from dm_control import mujoco
from dm_control.suite import common
import dm_control.suite.swimmer as swimmer
from lxml import etree
import numpy as np
import PIL.Image
import random
import matplotlib.pyplot as plt
import os
import argparse

DEFAULT_DENSITY = 0.01
DEFAULT_OBSTACLE_SIZE = 0.05
assets_dir = "/home/rdkeller/zebrafish_agent/zfa_rl_agent/core/environments/"
ASSETS = {}
for key, value in common.ASSETS.items():
    new_key = assets_dir + key
    ASSETS[new_key] = value
ASSETS

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


