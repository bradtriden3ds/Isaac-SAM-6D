import argparse
import glob
import logging
import os
from omegaconf import OmegaConf
import torch
from PIL import Image
import numpy as np
import imageio
import trimesh
import cv2
import distinctipy
from segment_anything.utils.amg import rle_to_mask
from skimage.feature import canny
from skimage.morphology import binary_dilation
import concurrent.futures

from Instance_Segmentation_Model.model.detector import Instance_Segmentation_Model as Detector
from Instance_Segmentation_Model.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from Instance_Segmentation_Model.model.dinov2 import CustomDINOv2
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad  
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from Instance_Segmentation_Model.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask


onboarding_config = OmegaConf.create({
    "rendering_type": "pbr",
    "reset_descriptors": False,
    "level_templates": 0,
})

matching_config = OmegaConf.create({
   "aggregation_function": "avg_5",
   "confidence_thresh": 0.2
})

post_processing_config = OmegaConf.create({
    "mask_post_processing": {
        "min_box_size": 0.05,   # relative to image size
        "min_mask_size": 3e-4,  # relative to image size
    },
    "nms_thresh": 0.25,
})


def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    args = parser.parse_args()

    # detector = Detector    

    sam = load_sam(
        model_type="vit_h",
        checkpoint_dir="./Instance_Segmentation_Model/checkpoints/segment-anything/",
    )

    segmentor_model = CustomSamAutomaticMaskGenerator(
        sam=sam,
        min_mask_region_area=0,
        points_per_batch=64,
        stability_score_thresh=0.97,
        box_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        segmentor_width_size=640,
        pred_iou_thresh=0.88,
    )

    descriptor_model = CustomDINOv2(
        model_name = "dinov2_vitl14",
        token_name = "x_norm_clstoken",
        descriptor_width_size=640,
        checkpoint_dir="./Instance_Segmentation_Model/checkpoints/dinov2/",
        image_size=224,
        chunk_size=16,
        validpatch_thresh=0.5,
    )

    model = Detector(
        segmentor_model=segmentor_model,
        descriptor_model=descriptor_model,
        onboarding_config=onboarding_config,
        matching_config=matching_config,
        post_processing_config=post_processing_config,
        log_interval=5,
        log_dir="./logs/sam",
        visible_thred=0.5,
        pointcloud_sample_num=2048,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")
        
    
    logging.info("Initializing template")
    template_dir = os.path.join(args.output_dir, 'templates')
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data

    mesh = trimesh.load_mesh(args.cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)

    # compute the geometric score
    depth_path = args.depth_path
    cam_path = args.cam_path
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    import time
    start_time = time.time()
    # run inference
    rgb = Image.open(args.rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)    

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    check_time_1 = time.time()
    logging.info(f"Time used for segmentation and descriptor extraction: {check_time_1 - start_time:.2f}s")

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    check_time_2 = time.time()
    logging.info(f"Time used for appearance matching: {check_time_2 - check_time_1:.2f}s")

    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )
    
    check_time_3 = time.time()
    logging.info(f"Time used for geometric matching: {check_time_3 - check_time_2:.2f}s")
    
    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
    detection_masks = force_binary_mask(detections.masks.cpu().data.numpy())
    detection_boxes = xyxy_to_xywh(detections.boxes.cpu().data.numpy())

    results = []
    for idx_det in range(len(detections.boxes)):
        result = {
            "scene_id": 0,
            "image_id": 0,
            "category_id": 0,
            "bbox": detection_boxes[idx_det].tolist(),
            "score": float(detections.scores[idx_det]),
            "time": 0,
            "segmentation": mask_to_rle(
                detection_masks[idx_det]
            ),
        }
        results.append(result)     
    
    # detections.to_numpy()
    save_path = f"{args.output_dir}/sam6d_results/detection_ism"
    # detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)

    end_time1 = time.time()
    logging.info(f"Total time: {end_time1 - start_time:.2f}s")

    import ipdb; ipdb.set_trace()

    # detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", results)
    vis_img = visualize(rgb, results, f"{args.output_dir}/sam6d_results/vis_ism_server2.png")
    vis_img.save(f"{args.output_dir}/sam6d_results/vis_ism_server2.png")