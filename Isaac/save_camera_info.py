import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import json

class CameraInfoSaver(Node):
    def __init__(self):
        super().__init__('camera_info_saver')
        # Subscribe once
        self.create_subscription(CameraInfo, '/camera_info', self.callback, 10)

    def callback(self, msg: CameraInfo):
        # Extract intrinsic matrix K
        cam_K = list(msg.k)  # msg.k is already a list of 9 floats

        # Depth scale, set default to 1.0 (you can modify if you have actual depth scale)
        depth_scale = 1.0

        # Prepare JSON dict
        camera_dict = {
            "cam_K": cam_K,
            "depth_scale": depth_scale
        }

        # Save to camera.json
        with open('camera.json', 'w') as f:
            json.dump(camera_dict, f, indent=2)

        self.get_logger().info("Saved camera.json")
        # Exit after saving
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoSaver()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
