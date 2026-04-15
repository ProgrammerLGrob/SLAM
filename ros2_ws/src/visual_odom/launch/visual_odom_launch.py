from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    visual_odom_node = Node(
        package='visual_odom',
        executable='visual_odom_node',
        name='visual_odom_node',
        output='screen'
    )
    
    rosbag_play_normal = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/media/sf_Projekt/20260324_Project_Bags/20260324_Project_Bags/pure_rotation_bag/rosbag2_2026_03_24-10_43_33_0.mcap','--clock'],
        output='screen'
    )

    return LaunchDescription([
        visual_odom_node,
        rosbag_play_normal,
    ])
