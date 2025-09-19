import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()

        # Subscribers
        self.depth_sub = self.create_subscription(Image, '/depth', self.depth_callback, 10)
        self.rgb_sub   = self.create_subscription(Image, '/rgb', self.rgb_callback, 10)

        self.depth_saved = False
        self.rgb_saved = False

    def depth_callback(self, msg):
        if not self.depth_saved:
            # Convert ROS2 depth image (32FC1) to numpy (meters)
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            # Convert to uint16 millimeters for saving
            depth_mm = (depth_img * 1000.0).astype(np.uint16)

            print("mean depth", np.mean(depth_mm))
            print("max depth", np.max(depth_mm))
            print("min depth", np.min(depth_mm))
            

            cv2.imwrite('depth.png', depth_mm)
            self.get_logger().info("Saved depth.png")
            self.depth_saved = True

    def rgb_callback(self, msg):
        if not self.rgb_saved:
            # Convert ROS2 RGB image to OpenCV format
            rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            cv2.imwrite('rgb.png', rgb_img)
            self.get_logger().info("Saved rgb.png")
            self.rgb_saved = True

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
