from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # launchの構成を示すLaunchDescription型の変数の定義
    return LaunchDescription([

    # publisher nodeを、"yolov8_detect_test_node"という名前で定義
    Node(
        package='ultralytics_ros',
        executable='detect_test_node.py',
        namespace='yolov8_detect',                 # namespace_app1というnamespaceを追加
        output = "screen",
        name = "yolo",
        #remappings=[('chatter', 'chatter_app1')]    # chatterトピックをchatter_app1トピックにremap
    ),

    ])

