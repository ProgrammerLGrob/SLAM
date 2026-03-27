from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    imgmatcher_node = Node(
        package='imgmatcher',
        executable='imgmatcher_node',
        name='image_matcher_node',
        output='screen'
    )
    
    rosbag_play_normal = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/media/sf_Projekt/20260324_Project_Bags/20260324_Project_Bags/pure_rotation_bag/rosbag2_2026_03_24-10_43_33_0.mcap','-l','--clock'],
        output='screen'
    )
    
    rosbag_play_remapped = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '/media/sf_Projekt/20260324_Project_Bags/20260324_Project_Bags/pure_rotation_bag/rosbag2_2026_03_24-10_43_33_0.mcap','-l','--clock','--remap','/serf01/nav_rgbd_1/rgb/image_raw:=/serf01/nav_rgbd_1/rgb/image_raw2'],
        output='screen'
    )

    return LaunchDescription([
        imgmatcher_node,
        rosbag_play_normal,
        rosbag_play_remapped,
    ])
