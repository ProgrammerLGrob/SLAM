from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    imglistener_node = Node(
        package='imglistener',
        executable='imglistener',
        name='image_subscriber',
        output='screen'
    )
    
    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/media/sf_Projekt/20260324_Project_Bags/20260324_Project_Bags/pure_rotation_3_bag/pure_rotation_3_bag/rosbag2_2026_03_31-12_04_52_0.mcap','-l','--clock'],
        output='screen'
    )

    return LaunchDescription([
        imglistener_node,
        rosbag_play,
    ])
