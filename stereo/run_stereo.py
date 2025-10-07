import os
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime
from PIL import Image
import time
from torchvision.transforms import ToPILImage

def normalize_image(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std

def pad_image(image):
    height, width = image.shape[-2:]
    pad_height = (((height // 32) + 1) * 32 - height) % 32
    pad_width = (((width // 32) + 1) * 32 - width) % 32
    pad = [pad_width // 2, pad_width - pad_width // 2, 0, pad_height]
    padded_image = F.pad(image, pad, mode='replicate')
    return padded_image, pad

def unpad_image(image, pad):
    height, width = image.shape[-2:]
    unpad_height = height - pad[3]
    unpad_width = width - pad[0] - pad[1]
    return image[:, :, :unpad_height, :unpad_width]

def load_image(file_path):
    image = Image.open(file_path).convert('RGB')
    image = image.resize((640, 480))  # Resize to 640x480
    image = torch.tensor(np.array(image).transpose((2, 0, 1)), dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def save_image(tensor, file_path):
    torch.save(tensor, file_path + ".pt")
    transform = ToPILImage()
    tensor = tensor.squeeze(0)  # Remove batch dimension
    pil_image = transform(tensor)
    pil_image.save(file_path)

def disparity_to_depth(disparity_map, focal_length, baseline):
    depth_map = (focal_length * baseline) / disparity_map
    return depth_map

def process_stereo_images(left_image_path, right_image_path, output_image_path, focal_length, baseline):
    start_time = time.time()

    # Load images
    load_start = time.time()
    left_image = load_image(left_image_path)
    right_image = load_image(right_image_path)
    load_end = time.time()
    print(f"Loading images took {load_end - load_start:.2f} seconds")

    # Normalize images
    normalize_start = time.time()
    left_image = normalize_image(left_image)
    right_image = normalize_image(right_image)
    normalize_end = time.time()
    print(f"Normalizing images took {normalize_end - normalize_start:.2f} seconds")

    # Pad images
    pad_start = time.time()
    left_image, left_pad = pad_image(left_image)
    right_image, right_pad = pad_image(right_image)
    pad_end = time.time()
    print(f"Padding images took {pad_end - pad_start:.2f} seconds")

    # Convert to numpy for ONNX inference
    convert_start = time.time()
    left_image_np = left_image.numpy()
    right_image_np = right_image.numpy()
    convert_end = time.time()
    print(f"Converting to numpy took {convert_end - convert_start:.2f} seconds")

    # ONNX inference
    createsession_start = time.time()
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 
    onnx_inputs = {"left_image": left_image_np, "right_image": right_image_np}
    session = onnxruntime.InferenceSession("/home/uxsimdeu/workspaces/Isaac-SAM-6D-main/stereo/models/deployable_foundation_stereo_l_dynamic.onnx", session_options, providers)
    createsession_end = time.time()
    print(f"ONNX create session took {createsession_end - createsession_start:.2f} seconds")
    inference_start = time.time()
    output = session.run(None, onnx_inputs)
    inference_end = time.time()
    print(f"ONNX inference took {inference_end - inference_start:.2f} seconds")

    # Unpad the output
    unpad_start = time.time()
    output_tensor = torch.tensor(output[0])
    output_tensor = unpad_image(output_tensor, left_pad)
    unpad_end = time.time()
    print(f"Unpadding output took {unpad_end - unpad_start:.2f} seconds")

    # Convert disparity to depth
    depth_start = time.time()
    depth_map = disparity_to_depth(output_tensor, focal_length, baseline)
    depth_end = time.time()
    print(f"Converting disparity to depth took {depth_end - depth_start:.2f} seconds")

    # Save the depth map as an image
    save_start = time.time()
    save_image(depth_map, output_image_path)
    save_end = time.time()
    print(f"Saving image took {save_end - save_start:.2f} seconds")

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

# Example usage
left_image_path = os.getenv('RGB_PATH')
right_image_path = os.getenv('RGB_PATH_RIGHT')
output_image_path = os.path.join(os.getenv('OUTPUT_DIR'), 'depth.png')
focal_length = 957.8115559493878  #focal length from isaac sim camera info
baseline = 150 #1.0  # baseline from isaac sim camera info in mm

if left_image_path and right_image_path:
    process_stereo_images(left_image_path, right_image_path, output_image_path, focal_length, baseline)
    print(f"Output depth map saved to {output_image_path}")
else:
    print("Environment variables RGB_PATH or RGB_PATH_RIGHT are not set.")
