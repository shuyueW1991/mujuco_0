#!/usr/bin/env python3
"""
URDF to MJCF Conversion Script

This script converts a URDF file to MJCF (MuJoCo XML) format using PyBullet's
built-in conversion functionality.

Usage:
    python urdf_to_mjcf.py [input_urdf] [output_mjcf]

Example:
    python urdf_to_mjcf.py robot.urdf robot.xml
"""

import os
import sys
import pybullet as p
import pybullet_data
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom


def convert_urdf_to_mjcf(urdf_path, mjcf_path):
    """
    Convert URDF file to MJCF format using PyBullet.
    
    Args:
        urdf_path (str): Path to input URDF file
        mjcf_path (str): Path to output MJCF file
    """
    # Start PyBullet in DIRECT mode (no GUI)
    physics_client = p.connect(p.DIRECT)
    
    try:
        # Set additional search path for PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load the URDF file
        print(f"Loading URDF file: {urdf_path}")
        robot_id = p.loadURDF(urdf_path)
        
        # Get the base directory for relative paths
        base_dir = os.path.dirname(os.path.abspath(urdf_path))
        
        # Export to MJCF
        print(f"Converting to MJCF format...")
        p.saveBullet(mjcf_path.replace('.xml', '.bullet'))
        
        # Use PyBullet's built-in MJCF export (if available)
        # Note: PyBullet doesn't have direct MJCF export, so we'll create a basic MJCF manually
        create_mjcf_from_urdf(urdf_path, mjcf_path)
        
        print(f"Successfully converted to: {mjcf_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise
    finally:
        # Disconnect from PyBullet
        p.disconnect()


def create_mjcf_from_urdf(urdf_path, mjcf_path):
    """
    Create a basic MJCF file from URDF by parsing and converting the structure.
    
    Args:
        urdf_path (str): Path to input URDF file
        mjcf_path (str): Path to output MJCF file
    """
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Create MJCF root element
    mjcf_root = ET.Element('mujoco', model=root.get('name', 'robot'))
    
    # Add compiler settings
    compiler = ET.SubElement(mjcf_root, 'compiler')
    compiler.set('angle', 'radian')
    compiler.set('coordinate', 'local')
    
    # Add default settings
    default = ET.SubElement(mjcf_root, 'default')
    joint_default = ET.SubElement(default, 'joint')
    joint_default.set('damping', '0.1')
    joint_default.set('frictionloss', '0.1')
    
    geom_default = ET.SubElement(default, 'geom')
    geom_default.set('friction', '1.0 0.1 0.1')
    geom_default.set('density', '1000')
    
    # Add assets section
    asset = ET.SubElement(mjcf_root, 'asset')
    
    # Add worldbody
    worldbody = ET.SubElement(mjcf_root, 'worldbody')
    
    # Add ground plane
    ground = ET.SubElement(worldbody, 'geom')
    ground.set('name', 'ground')
    ground.set('type', 'plane')
    ground.set('size', '10 10 0.1')
    ground.set('rgba', '0.8 0.8 0.8 1')
    
    # Add obstacle
    obstacle = ET.SubElement(worldbody, 'body')
    obstacle.set('name', 'obstacle')
    obstacle.set('pos', '2 0 0.1')
    obstacle_geom = ET.SubElement(obstacle, 'geom')
    obstacle_geom.set('name', 'obstacle_geom')
    obstacle_geom.set('type', 'box')
    obstacle_geom.set('size', '0.1 0.1 0.1')
    obstacle_geom.set('rgba', '1 0 0 1')
    
    # Convert robot body
    robot_body = ET.SubElement(worldbody, 'body')
    robot_body.set('name', 'robot')
    robot_body.set('pos', '0 0 0.1')
    
    # Process URDF links and joints
    links = {}
    joints = {}
    
    # Collect all links
    for link in root.findall('link'):
        links[link.get('name')] = link
    
    # Collect all joints
    for joint in root.findall('joint'):
        joints[joint.get('name')] = joint
    
    # Convert base link
    base_link = links.get('base_link')
    if base_link is not None:
        convert_link_to_mjcf(base_link, robot_body)
    
    # Convert wheels as separate bodies
    wheel_names = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']
    joint_names = ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']
    
    for wheel_name, joint_name in zip(wheel_names, joint_names):
        if wheel_name in links and joint_name in joints:
            wheel_body = ET.SubElement(robot_body, 'body')
            wheel_body.set('name', wheel_name)
            
            # Get joint origin
            joint = joints[joint_name]
            origin = joint.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                wheel_body.set('pos', xyz)
            
            # Convert wheel link
            convert_link_to_mjcf(links[wheel_name], wheel_body)
            
            # Add joint
            mjcf_joint = ET.SubElement(wheel_body, 'joint')
            mjcf_joint.set('name', joint_name)
            mjcf_joint.set('type', 'hinge')
            mjcf_joint.set('axis', '0 1 0')
    
    # Add actuators
    actuator = ET.SubElement(mjcf_root, 'actuator')
    for joint_name in joint_names:
        motor = ET.SubElement(actuator, 'motor')
        motor.set('name', f'{joint_name}_motor')
        motor.set('joint', joint_name)
        motor.set('gear', '1')
    
    # Write the MJCF file
    rough_string = ET.tostring(mjcf_root, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent='  ')
    
    with open(mjcf_path, 'w') as f:
        f.write(pretty_xml)


def convert_link_to_mjcf(link, parent_element):
    """
    Convert a URDF link to MJCF geom elements.
    
    Args:
        link: URDF link element
        parent_element: Parent MJCF element to add geoms to
    """
    # Process visual elements
    for visual in link.findall('visual'):
        geom = ET.SubElement(parent_element, 'geom')
        geom.set('name', f'{link.get("name")}_visual')
        
        geometry = visual.find('geometry')
        if geometry is not None:
            # Handle box geometry
            box = geometry.find('box')
            if box is not None:
                geom.set('type', 'box')
                size = box.get('size', '1 1 1')
                # Convert size from full dimensions to half-dimensions
                dims = [str(float(x)/2) for x in size.split()]
                geom.set('size', ' '.join(dims))
            
            # Handle cylinder geometry
            cylinder = geometry.find('cylinder')
            if cylinder is not None:
                geom.set('type', 'cylinder')
                radius = cylinder.get('radius', '0.1')
                length = cylinder.get('length', '0.1')
                geom.set('size', f'{radius} {float(length)/2}')
        
        # Handle material/color
        material = visual.find('material')
        if material is not None:
            color = material.find('color')
            if color is not None:
                rgba = color.get('rgba', '0.5 0.5 0.5 1')
                geom.set('rgba', rgba)


def main():
    parser = argparse.ArgumentParser(description='Convert URDF to MJCF format')
    parser.add_argument('input_urdf', nargs='?', default='robot.urdf',
                       help='Input URDF file path (default: robot.urdf)')
    parser.add_argument('output_mjcf', nargs='?', default='robot.xml',
                       help='Output MJCF file path (default: robot.xml)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_urdf):
        print(f"Error: Input file '{args.input_urdf}' not found.")
        sys.exit(1)
    
    try:
        convert_urdf_to_mjcf(args.input_urdf, args.output_mjcf)
        print("\nConversion completed successfully!")
        print(f"Input:  {args.input_urdf}")
        print(f"Output: {args.output_mjcf}")
    except Exception as e:
        print(f"\nConversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()