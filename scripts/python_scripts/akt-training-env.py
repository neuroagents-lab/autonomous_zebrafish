from dm_control import mujoco
from dm_control.suite import common
from lxml import etree
import numpy as np
import PIL.Image
import random
import matplotlib.pyplot as plt
import os
import argparse

# Configuration
DEFAULT_INPUT_PATH = '../../zfa_rl_agent/mujoco_bodies/swimmer_random_obstacles.xml'
DEFAULT_OUTPUT_PATH = "../../zfa_rl_agent/mujoco_bodies/swimmer_random_obstacles_new.xml"
DEFAULT_DENSITY = 0.01
DEFAULT_OBSTACLE_SIZE = 0.05

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate swimmer model with random obstacles")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input XML file path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output XML file path")
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY, help="Obstacle density")
    parser.add_argument("--obstacle-size", type=float, default=DEFAULT_OBSTACLE_SIZE, help="Obstacle size")
    return parser.parse_args()

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

def _make_model(n_bodies, positions, num_obstacles=0):
  """Generates an xml string defining a swimmer with `n_bodies` bodies.
  
  Arguments:
  n_bodies: number of bodies in the swimmer
  positions: list of obstacle positions
  num_obstacles: number of obstacles to place
  
  Returns:
  XML string defining the swimmer"""
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
    'file': '../mujoco_bodies/common/OutdoorGrassFloorD.png'
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


  # Remove target from the model
  target = mjcf.find('.//geom[@name="target"]')
  target_light = mjcf.find('.//light[@name="target_light"]')
  target_sensor = mjcf.find('.//sensor/framepos[@name="target_pos"][@objname="target"]')
  if target is not None:
      target.getparent().remove(target) 
  if target_light is not None:
      target_light.getparent().remove(target_light) 
  if target_sensor is not None:
      target_sensor.getparent().remove(target_sensor)

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

  # Obstacles
  height = 0.1
  for obstacle in range(num_obstacles):
    rand_pos = str(positions[obstacle][0]) + ' ' + str(positions[obstacle][1]) + ' ' +  str(height/2)
    obstacle = etree.Element('geom', {
    'name': f'obstacle_{obstacle}',
    'type': 'cylinder',
    'mass': '1',
    'size': '0.05 0.1', # Radius and height of the cylinder
    'rgba': '0 1 0 0.5',           
    'pos': rand_pos             
    })
    mjcf.find('./worldbody').append(obstacle)
  # Move tracking cameras further away from the swimmer according to its length.
  cameras = mjcf.findall('./worldbody/body/camera')
  scale = n_bodies / 6.0
  for cam in cameras:
    if cam.get('mode') == 'trackcom':
      old_pos = cam.get('pos').split(' ')
      new_pos = ' '.join([str(float(dim) * scale) for dim in old_pos])
      cam.set('pos', new_pos)

  return etree.tostring(mjcf, pretty_print=True)

def check_overlap(new_pos, obstacle_positions, min_dist):
    """Check if a new position overlaps with existing obstacles.

    Arguments:
    new_pos: [x, y] position of the new obstacle
    obstacle_positions: list of [x, y] positions of existing obstacles
    min_dist: minimum distance between

    Returns:
    True if the new position overlaps with existing obstacles, False otherwise
    """
    return any(np.linalg.norm(np.array(new_pos) - np.array(pos)) < min_dist for pos in obstacle_positions)

def get_element_position(root, body_name):
    """Get the position of an element from the XML tree.
    
    Arguments:
    root: root of the XML tree
    body_name: name of the body element to find
    
    Returns:
    Position of the body element if found, None otherwise
    """
    body_element = root.find(f".//body[@name='{body_name}']")
    return body_element.get("pos") if body_element is not None else None

def count_segments(root):
    """Count the number of segments in the swimmer model.
    
    Arguments:
    root: root of the XML tree
    
    Returns:
    Number of segments in the swimmer model
    """
    num_segments = 0
    while get_element_position(root, f"segment_{num_segments}") is not None:
        num_segments += 1
    return num_segments

def calculate_parameters(physics, size_body, obstacle_size, density):
    """Calculate parameters for obstacle placement.
    
    Arguments:
    physics: mujoco.Physics object
    size_body: size of the body
    obstacle_size: size of the obstacles
    density: obstacle density
    
    Returns:
    x_min_neg, x_max_neg, x_min_pos, x_max_pos, y_min_neg, y_max_neg, y_min_pos, y_max_pos, n_obstacles
    """
    ground_size = physics.named.model.geom_size['ground'] * 2 
    ground_area = ground_size[0] * ground_size[1] 
    obstacle_area = obstacle_size ** 2 

    x_min_neg, x_max_neg = -(ground_size[0]/2), -(size_body + obstacle_size/2)
    x_min_pos, x_max_pos = size_body + obstacle_size/2, (ground_size[0])/2
    y_min_neg, y_max_neg = -(ground_size[1]/2), -(size_body + obstacle_size/2)
    y_min_pos, y_max_pos = size_body + obstacle_size/2, (ground_size[1]/2)

    inside_area = (x_max_neg - x_min_pos) * (y_max_neg - y_min_pos)
    desired_covered_area = (ground_area - inside_area) * density
    n_obstacles = int(desired_covered_area / obstacle_area)

    return x_min_neg, x_max_neg, x_min_pos, x_max_pos, y_min_neg, y_max_neg, y_min_pos, y_max_pos, n_obstacles

def place_obstacles(n_obstacles, x_min_neg, x_max_neg, x_min_pos, x_max_pos, 
                    y_min_neg, y_max_neg, y_min_pos, y_max_pos, min_distance):
    """Place obstacles randomly in the designated area.
    
    Arguments:
    n_obstacles: number of obstacles to place
    x_min_neg, x_max_neg, x_min_pos, x_max_pos, y_min_neg, y_max_neg, y_min_pos, y_max_pos: boundaries of the area
    min_distance: minimum distance between obstacles

    Returns:
    List of obstacle positions
    """
    obstacle_positions = []
    for _ in range(n_obstacles):
        while True:
            x_or_y = random.choice(['x', 'y'])
            if x_or_y == 'x':
                x_pos = random.uniform(x_min_neg, x_max_pos)
                if x_min_pos > x_pos > x_max_neg:
                    y_range = random.choice([(y_min_neg, y_max_neg), (y_min_pos, y_max_pos)])
                    y_pos = random.uniform(y_range[0], y_range[1])
                else:
                    y_pos = random.uniform(y_min_neg, y_max_pos)
            else:
                y_pos = random.uniform(y_min_neg, y_max_pos)
                if y_min_pos > y_pos > y_max_neg:
                    x_range = random.choice([(x_min_neg, x_max_neg), (x_min_pos, x_max_pos)])
                    x_pos = random.uniform(x_range[0], x_range[1])
                else:
                    x_pos = random.uniform(x_min_neg, x_max_pos)
            
            if not check_overlap([x_pos, y_pos], obstacle_positions, min_distance):
                obstacle_positions.append([x_pos, y_pos])
                break
    return obstacle_positions

def main():
    args = parse_arguments()

    try:
        tree = etree.parse(args.input)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing input XML: {e}")
        return

    num_segments = count_segments(root)
    size_body = 0.04 + 0.1 * num_segments
    print(f"Size of body: {size_body}")

    try:
        physics = mujoco.Physics.from_xml_path(args.input)
    except Exception as e:
        print(f"Error loading physics model: {e}")
        return

    x_min_neg, x_max_neg, x_min_pos, x_max_pos, y_min_neg, y_max_neg, y_min_pos, y_max_pos, n_obstacles = calculate_parameters(
        physics, size_body, args.obstacle_size, args.density)

    obstacle_positions = place_obstacles(n_obstacles, x_min_neg, x_max_neg, x_min_pos, x_max_pos, 
                                         y_min_neg, y_max_neg, y_min_pos, y_max_pos, args.obstacle_size)

    try:
        xml_string = _make_model(num_segments, obstacle_positions, len(obstacle_positions))
        with open(args.output, "wb") as f:
            f.write(xml_string)
        print(f"Swimmer model saved to {args.output}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

if __name__ == "__main__":
    main()
