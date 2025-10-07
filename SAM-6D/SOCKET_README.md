# SAM-6D Socket Server

This is a socket-based implementation of the SAM-6D server that replaces the FastAPI version for better performance and direct byte communication.

## Features

- **Direct Socket Communication**: Bypasses HTTP overhead for faster data transfer
- **Binary Data Support**: Handles RGB and depth image bytes directly
- **Concurrent Connections**: Supports multiple simultaneous clients
- **Simple Protocol**: Uses pickle serialization with length prefixes
- **Error Handling**: Robust error handling and connection management

## Usage

### Starting the Server

```bash
# Basic usage (localhost:8000)
python start_server_socket.py

# Custom host and port
python start_server_socket.py --host 0.0.0.0 --port 9000
```

### Using the Client

```bash
# Example with provided test images
python socket_client_example.py --rgb ./Data/Example6/isaacsim_camera_capture_19_left.png --depth ./Data/Example6/depth_map.png

# Custom threshold and no visualization
python socket_client_example.py --rgb /path/to/rgb.png --depth /path/to/depth.png --threshold 0.7 --no-viz
```

## Protocol

The socket protocol uses pickle serialization with a 4-byte length prefix:

1. **Request Format**:
```python
{
    'action': 'sam6d_inference',  # or 'ping', 'shutdown'
    'rgb_bytes': <RGB image bytes>,
    'depth_bytes': <Depth image bytes>,
    'det_score_thresh': 0.5,      # Detection threshold (optional)
    'visualize': True             # Enable visualization (optional)
}
```

2. **Response Format**:
```python
{
    'status': 'success',  # or 'error'
    'data': {
        'rotation': [...],      # List of rotation matrices
        'translation': [...],   # List of translation vectors
        'pose_scores': [...]    # List of confidence scores
    },
    'message': '...'  # Error message (if status is 'error')
}
```

## Client Implementation

### Basic Python Client

```python
import socket
import struct
import pickle

class SAM6DClient:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
    
    def send_data(self, data):
        serialized = pickle.dumps(data)
        length = struct.pack('!I', len(serialized))
        self.socket.sendall(length + serialized)
    
    def receive_data(self):
        # Receive length prefix
        length_data = b''
        while len(length_data) < 4:
            chunk = self.socket.recv(4 - len(length_data))
            length_data += chunk
        length = struct.unpack('!I', length_data)[0]
        
        # Receive actual data
        data = b''
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            data += chunk
        return pickle.loads(data)
    
    def run_inference(self, rgb_path, depth_path):
        with open(rgb_path, 'rb') as f:
            rgb_bytes = f.read()
        with open(depth_path, 'rb') as f:
            depth_bytes = f.read()
        
        request = {
            'action': 'sam6d_inference',
            'rgb_bytes': rgb_bytes,
            'depth_bytes': depth_bytes
        }
        
        self.send_data(request)
        return self.receive_data()
```

## Environment Setup

Make sure to set the required environment variables:

```bash
export CAD_PATH="/path/to/your/cad/model.ply"
export OUTPUT_DIR="/path/to/output/directory"
export CAM_PATH="/path/to/camera/config.json"
```

## Advantages over FastAPI

1. **Performance**: Direct socket communication eliminates HTTP overhead
2. **Memory Efficiency**: Binary data transfer without base64 encoding
3. **Simplicity**: No web framework dependencies
4. **Flexibility**: Custom protocol optimized for SAM-6D workflow
5. **Concurrent Processing**: Thread-based handling of multiple clients

## Supported Actions

- `sam6d_inference`: Run 6D pose estimation on RGB/depth images
- `ping`: Health check (returns 'pong')
- `shutdown`: Gracefully shutdown the server

## Error Handling

The server includes comprehensive error handling:
- Connection drops are handled gracefully
- Invalid requests return error responses
- Model loading errors are logged
- Resource cleanup on shutdown

## Dependencies

Same as the original SAM-6D implementation, plus:
- `socket` (built-in)
- `threading` (built-in) 
- `struct` (built-in)
- `pickle` (built-in)

No additional dependencies are required for the socket implementation.
