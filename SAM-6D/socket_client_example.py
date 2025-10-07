#!/usr/bin/env python3
"""
Example client for SAM-6D Socket Server

This script demonstrates how to connect to the SAM-6D socket server and send
RGB/depth image data for 6D pose estimation.
"""

import socket
import struct
import pickle
import argparse
import logging

logging.basicConfig(level=logging.INFO)


class SAM6DClient:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.socket = None
    
    def connect(self):
        """Connect to the SAM-6D server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logging.info(f"Connected to SAM-6D server at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to server: {str(e)}")
            return False
    
    def send_data(self, data):
        """Send data to the server"""
        serialized = pickle.dumps(data)
        length = struct.pack('!I', len(serialized))
        self.socket.sendall(length + serialized)
    
    def receive_data(self):
        """Receive data from the server"""
        # First, receive the length of the message
        length_data = b''
        while len(length_data) < 4:
            chunk = self.socket.recv(4 - len(length_data))
            if not chunk:
                raise ConnectionError("Connection closed")
            length_data += chunk
        
        length = struct.unpack('!I', length_data)[0]
        
        # Now receive the actual data
        data = b''
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
            
        return pickle.loads(data)
    
    def ping(self):
        """Send a ping to check if server is alive"""
        try:
            request = {'action': 'ping'}
            self.send_data(request)
            response = self.receive_data()
            return response
        except Exception as e:
            logging.error(f"Ping failed: {str(e)}")
            return None
    
    def run_sam6d_inference(self, rgb_image_path, depth_image_path, 
                           det_score_thresh=0.5, visualize=True):
        """
        Run SAM-6D inference on RGB and depth images
        
        Args:
            rgb_image_path: Path to RGB image file
            depth_image_path: Path to depth image file
            det_score_thresh: Detection score threshold (default: 0.5)
            visualize: Whether to generate visualization (default: True)
            
        Returns:
            Dictionary with rotation, translation, and pose_scores
        """
        try:
            # Read image files as bytes
            with open(rgb_image_path, 'rb') as f:
                rgb_bytes = f.read()
            
            with open(depth_image_path, 'rb') as f:
                depth_bytes = f.read()
            
            # Prepare request
            request = {
                'action': 'sam6d_inference',
                'rgb_bytes': rgb_bytes,
                'depth_bytes': depth_bytes,
                'det_score_thresh': det_score_thresh,
                'visualize': visualize
            }
            
            logging.info("Sending SAM-6D inference request...")
            self.send_data(request)
            
            logging.info("Waiting for response...")
            response = self.receive_data()
            
            return response
            
        except Exception as e:
            logging.error(f"SAM-6D inference failed: {str(e)}")
            return None
    
    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            self.socket.close()
            logging.info("Disconnected from server")


def main():
    """Example usage of the SAM-6D client"""
    parser = argparse.ArgumentParser(description='SAM-6D Socket Client Example')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to depth image')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Detection score threshold')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create client and connect
    client = SAM6DClient(host=args.host, port=args.port)
    
    if not client.connect():
        return
    
    try:
        # Test ping
        logging.info("Testing connection...")
        ping_response = client.ping()
        if ping_response:
            logging.info(f"Server response: {ping_response}")
        else:
            logging.error("Ping failed")
            return
        
        # Run SAM-6D inference
        result = client.run_sam6d_inference(
            rgb_image_path=args.rgb,
            depth_image_path=args.depth,
            det_score_thresh=args.threshold,
            visualize=not args.no_viz
        )
        
        if result and result.get('status') == 'success':
            data = result['data']
            logging.info("SAM-6D inference completed successfully!")
            logging.info(f"Number of detections: {len(data['rotation'])}")
            
            # Print results
            for i, (rot, trans, score) in enumerate(zip(
                data['rotation'], data['translation'], data['pose_scores']
            )):
                logging.info(f"Detection {i+1}:")
                logging.info(f"  Rotation: {rot}")
                logging.info(f"  Translation: {trans}")
                logging.info(f"  Score: {score}")
                
        elif result:
            logging.error(f"Inference failed: {result.get('message', 'Unknown error')}")
        else:
            logging.error("No response received")
            
    except KeyboardInterrupt:
        logging.info("Client interrupted by user")
    except Exception as e:
        logging.error(f"Client error: {str(e)}")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
