from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import yaml

from visual_odom.constants import (
    RGB_IMAGE_TOPIC,
    DEPTH_IMAGE_TOPIC,
    WHEEL_ODOMETRY_TOPIC,
    FILTERED_ODOMETRY_TOPIC,
    IMU_ODOMETRY_TOPIC,
    TF_STATIC_TOPIC,
    PERCENTAGE_OF_PLAY_SPEED
)


def generate_launch_description():
    visual_odom_share = get_package_share_directory('visual_odom')
    config_file = os.path.join(visual_odom_share, 'config', 'param.yaml')

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    rosbag_path = config.get('visual_odom_node', {}).get('ros__parameters', {}).get('rosbag_path', '')
    rosbag_path = os.path.join(visual_odom_share, '../../../../../' + rosbag_path)
    rviz_config = os.path.join(visual_odom_share, '../../../../src/visual_odom/rviz', 'rviz2_config.rviz')
    
    visual_odom_node = Node(
        package='visual_odom',
        executable='visual_odom_node',
        name='visual_odom_node',
        output='screen',
        parameters=[config_file]
    )

    rosbag_play_normal = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', rosbag_path,
            '--clock', '-r', str(PERCENTAGE_OF_PLAY_SPEED),
            '--topics',
            RGB_IMAGE_TOPIC,
            DEPTH_IMAGE_TOPIC,
            WHEEL_ODOMETRY_TOPIC,
            FILTERED_ODOMETRY_TOPIC,
            IMU_ODOMETRY_TOPIC,
            TF_STATIC_TOPIC
        ],
        output='screen'
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        visual_odom_node,
        rosbag_play_normal,
        rviz2,
    ])