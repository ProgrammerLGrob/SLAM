from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    visual_odom_share = get_package_share_directory('visual_odom')
    config_file = os.path.join(visual_odom_share, 'config', 'param.yaml')
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    rosbag_path = config.get('visual_odom_node', {}).get('ros__parameters', {}).get('rosbag_path', '')
    
    visual_odom_node = Node(
        package='visual_odom',
        executable='visual_odom_node',
        name='visual_odom_node',
        output='screen',
        parameters=[config_file]
    )
    
    rosbag_play_normal = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--clock', '--topics',
             '/serf01/nav_rgbd_1/rgb/image_raw',
             '/serf01/nav_rgbd_1/depth/image_raw',
             '/serf01/odometry/wheel',
             '/serf01/odometry/filtered',
             '/serf01/odometry/imu',
             '/tf_static'],
        output='screen'
    )

    rviz2 = ExecuteProcess(
        cmd=['rviz2'],
        output='screen'
    )
    
    return LaunchDescription([
        visual_odom_node,
        rosbag_play_normal,
        rviz2,
    ])