import torch
import os
from PIL import Image
from torchvision.transforms import ToPILImage

def load_image(file_path):
    image = Image.open(file_path).convert('RGB')
    image = image.resize((640, 480))  # Resize to 640x480
    return image

# tensor to image conversion
def tensor_to_image():
    output_image_path : str = os.path.join(os.getenv('OUTPUT_DIR'),'depth.png')
    tensor = torch.load(output_image_path + ".pt")

    transform = ToPILImage()
    tensor = tensor.squeeze(0)  # Remove batch dimension
    pil_image = transform(tensor)
    pil_image.save(output_image_path)


left_image_path : str = os.getenv('RGB_PATH')
img = load_image(left_image_path)
img.save(left_image_path + "_small.png")
