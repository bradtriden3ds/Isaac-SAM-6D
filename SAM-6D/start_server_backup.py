import argparse
import glob
import json
import logging
import os
import gorilla
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import imageio
import trimesh
import time


from Instance_Segmentation_Model.model.detector import Instance_Segmentation_Model as Detector
from Instance_Segmentation_Model.model.sam import CustomSamAutomaticMaskGenerator, load_sam
from Instance_Segmentation_Model.model.dinov2 import CustomDINOv2
from Instance_Segmentation_Model.utils.bbox_utils import CropResizePad  
from Instance_Segmentation_Model.model.utils import Detections, convert_npz_to_json
from Instance_Segmentation_Model.utils.inout import load_json, save_json_bop23
from Instance_Segmentation_Model.utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from Instance_Segmentation_Model.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask

from Pose_Estimation_Model.model.pose_estimation_model import Net as Estimator
from Pose_Estimation_Model.utils.draw_utils import non_max_suppression

import trimesh

from configs import onboarding_config, matching_config, post_processing_config, pose_estimation_config, pose_estimation_test_config
from utils import batch_input_data, visualize_image_segmentation, visualize_pose_estimation, get_templates, get_test_data, mask_to_rle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import dotenv
dotenv.load_dotenv(override=True)

def load_detector():
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

    detector = Detector(
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

    
    detector.descriptor_model.model = detector.descriptor_model.model.to(device)
    detector.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(detector.segmentor_model, "predictor"):
        detector.segmentor_model.predictor.model = (
            detector.segmentor_model.predictor.model.to(device)
        )
    else:
        detector.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Loading detector model to {device} done!")

    return detector

def init_templates(detector, cad_path: str, output_dir: str):
    """
    Initialize the templates for the detector.
    ::param detector: The detector model.
    ::param cad_path: The path to the CAD model (in mm).
    ::param output_dir: The path to the output directory where the templates are stored.
    """
    logging.info("Initializing template")
    template_dir = os.path.join(output_dir, 'templates')
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

    detector.ref_data = {}
    detector.ref_data["descriptors"] = detector.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    detector.ref_data["appe_descriptors"] = detector.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    detector.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)

    # to compute the geometric score

    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    detector.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]



def load_estimator():
    estimator = Estimator(pose_estimation_config).to(device)
    estimator.eval()
    checkpoint = os.path.join("./Pose_Estimation_Model", 'checkpoints', 'sam-6d-pem-base.pth')
    print("=> loading checkpoint ...", checkpoint)
    gorilla.solver.load_checkpoint(model=estimator, filename=checkpoint)

    return estimator

from fastapi import FastAPI, Query
from contextlib import asynccontextmanager 

sam6d_models = {"detector": None, "estimator": None, "batch": None, "templates": {}}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    sam6d_models["detector"] = load_detector()
    sam6d_models["estimator"] = load_estimator()
    init_templates(sam6d_models["detector"], os.getenv("CAD_PATH"), os.getenv("OUTPUT_DIR"))
    sam6d_models["batch"] = batch_input_data(os.getenv("DEPTH_PATH"), os.getenv("CAM_PATH"), device)
    sam6d_models["templates"] = get_templates(os.path.join(os.getenv("OUTPUT_DIR"), 'templates'), pose_estimation_test_config)
    yield
    # Clean up the ML models and release the resources
    del sam6d_models["detector"]
    del sam6d_models["estimator"]
    del sam6d_models["batch"]
    torch.cuda.empty_cache()

# Create app
app = FastAPI(lifespan=lifespan)

# Define a route
@app.get("/")
def read_root():
    return {"message": "Running SAM-6D server..."}


@app.get("/sam6d")
def run_sam6d(
    rgb_path: str = "./Data/Example6/isaacsim_camera_capture_19_left.png",
    depth_path: str = "./Data/Example6/depth_map.png",
    det_score_thresh: float = 0.5,
):
    output_dir = os.getenv("OUTPUT_DIR")
    cam_path = os.getenv("CAM_PATH")
    cad_path = os.getenv("CAD_PATH")

    start_time = time.time()

    detector = sam6d_models["detector"]
    batch = sam6d_models["batch"]
    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = detector.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    query_decriptors, query_appe_descriptors = detector.descriptor_model.forward(np.array(rgb), detections)    

    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = detector.compute_semantic_score(query_decriptors)

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= detector.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    image_uv = detector.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = detector.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=detector.visible_thred
        )
    
    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
    detection_masks = force_binary_mask(detections.masks.cpu().data.numpy())
    detection_boxes = xyxy_to_xywh(detections.boxes.cpu().data.numpy())

    results = []
    for idx_det in range(len(detections.boxes)):
        if detections.scores[idx_det] < det_score_thresh:
            continue
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
    output_dir = os.getenv("OUTPUT_DIR")
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    # detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)

    # detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", results)
    vis_img = visualize_image_segmentation(rgb, results, f"{output_dir}/sam6d_results/vis_ism_server3.png")
    vis_img.save(f"{output_dir}/sam6d_results/vis_ism_server3.png")

    image_segmentation_time_cost = str(time.time() - start_time)

    # Pose estimation
    estimator = sam6d_models["estimator"]
    all_tem, all_tem_pts, all_tem_choose = sam6d_models["templates"]

    with torch.no_grad():
        all_tem_pts, all_tem_feat = estimator.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    seg_path = os.path.join(output_dir, 'sam6d_results', 'detection_ism.json')
    input_data, img, whole_pts, model_points, detections = get_test_data(
        rgb_path, depth_path, cam_path, cad_path, seg_path, 
        det_score_thresh, pose_estimation_test_config
    )
    ninstance = input_data['pts'].size(0)

    print("=> running model ...")
    with torch.no_grad():
        input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
        input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)

        # get output in batch to avoid memory error
        bs = 4
        n_batch = int(np.ceil(ninstance/bs))

        out = {'pred_pose_score': [], 'score': [], 'pred_R': [], 'pred_t': []}
        for j in range(n_batch):
            start_idx = j * bs
            end_idx = ninstance if j == n_batch-1 else (j+1) * bs
            batch_input_data = {key: input_data[key][start_idx:end_idx] for key in input_data}
            batch_out = estimator(batch_input_data)
            for key in out:
                out[key].append(batch_out[key])
        for key in out:
            out[key] = torch.cat(out[key], dim=0)
            
        # out = model(input_data)
    
    if 'pred_pose_score' in out.keys():
        pose_scores = out['pred_pose_score'] * out['score']
    else:
        pose_scores = out['score']
    pose_scores = pose_scores.detach().cpu().numpy()
    pred_rot = out['pred_R'].detach().cpu().numpy()
    pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

    print("=> applying Non-Maximum Suppression...")
    
    # First, update all detections with pose estimation results
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    for idx, det in enumerate(detections):
        detections[idx]['score'] = float(pose_scores[idx])
        detections[idx]['R'] = list(pred_rot[idx].tolist())
        detections[idx]['t'] = list(pred_trans[idx].tolist())
        detections[idx]['s'] = list(scale.tolist())
    
    # Apply Non-Maximum Suppression to filter overlapping detections
    print(f"=> Before NMS: {len(detections)} detections")

    # TODO: remove this range(len(detections)) # 
    nms_indices = non_max_suppression(detections, iou_threshold=0.2)
    print(f"=> After NMS: {len(nms_indices)} detections")
    
    # Filter detections, predictions, and other data based on NMS results
    filtered_detections = [detections[i] for i in nms_indices]
    filtered_pred_rot = pred_rot[nms_indices]
    filtered_pred_trans = pred_trans[nms_indices]
    filtered_pose_scores = pose_scores[nms_indices]

    # Save filtered results
    print("=> saving filtered results ...")
    os.makedirs(f"{output_dir}/sam6d_results", exist_ok=True)
    with open(os.path.join(f"{output_dir}/sam6d_results", 'detection_pem.json'), "w") as f:
        json.dump(filtered_detections, f)
    
    # Update detections for visualization
    detections = filtered_detections

    print("=> visualizating ...")
    
    for idx in range(len(detections)):
        K = np.expand_dims(input_data['K'].detach().cpu().numpy()[idx], axis=0)
        save_path = os.path.join(f"{output_dir}/sam6d_results", 'vis_pem_server3'+str(idx)+'.png')
        vis_img = visualize_pose_estimation(img, 
            np.expand_dims(filtered_pred_rot[idx], axis=0), 
            np.expand_dims(filtered_pred_trans[idx], axis=0), 
            model_points*1000, K, save_path,
            score=detections[idx]['score']
            )
        vis_img.save(save_path)

    total_time = str(time.time() - start_time)

    return {"message": "SAM-6D inference completed.", "image_segmentation_time_cost": image_segmentation_time_cost, "total_time": total_time}


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--det_score_thresh", type=float, default=0.5, help="Threshold for detection score")
    # create a boolean flag (default true) to indicate whether to use image segmentation or not
    parser.add_argument("--use_image_segmentation", action='store_false', help="Whether to use image segmentation")
    parser.add_argument("--use_pose_estimation", action='store_false', help="Whether to use pose estimation")
    args = parser.parse_args()


    # Instance Segmentation
    if args.use_image_segmentation:
        # init detector model
        detector = load_detector()
        
        # init templates
        init_templates(detector, args.cad_path, args.output_dir)
        
        # init input data
        batch = batch_input_data(args.depth_path, args.cam_path, device)
        
        import time
        start_time = time.time()
        # run inference
        rgb = Image.open(args.rgb_path).convert("RGB")
        detections = detector.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        query_decriptors, query_appe_descriptors = detector.descriptor_model.forward(np.array(rgb), detections)    

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = detector.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        check_time_1 = time.time()
        logging.info(f"Time used for segmentation and descriptor extraction: {check_time_1 - start_time:.2f}s")

        # compute the appearance score
        appe_scores, ref_aux_descriptor= detector.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        check_time_2 = time.time()
        logging.info(f"Time used for appearance matching: {check_time_2 - check_time_1:.2f}s")

        image_uv = detector.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = detector.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=detector.visible_thred
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

        # detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", results)
        vis_img = visualize_image_segmentation(rgb, results, f"{args.output_dir}/sam6d_results/vis_ism_server2.png")
        vis_img.save(f"{args.output_dir}/sam6d_results/vis_ism_server2.png")


    # Pose Estimation
    if args.use_pose_estimation:
        print("=> loading estimator model ...")
        estimator = load_estimator()

        print("=> extracting templates ...")
        tem_path = os.path.join(args.output_dir, 'templates')
        all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, pose_estimation_test_config)
        with torch.no_grad():
            all_tem_pts, all_tem_feat = estimator.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

        seg_path = os.path.join(args.output_dir, 'sam6d_results', 'detection_ism.json')
        input_data, img, whole_pts, model_points, detections = get_test_data(
            args.rgb_path, args.depth_path, args.cam_path, args.cad_path, seg_path, 
            args.det_score_thresh, pose_estimation_test_config
        )
        ninstance = input_data['pts'].size(0)

        print("=> running model ...")
        with torch.no_grad():
            input_data['dense_po'] = all_tem_pts.repeat(ninstance,1,1)
            input_data['dense_fo'] = all_tem_feat.repeat(ninstance,1,1)

            # get output in batch to avoid memory error
            bs = 4
            n_batch = int(np.ceil(ninstance/bs))

            out = {'pred_pose_score': [], 'score': [], 'pred_R': [], 'pred_t': []}
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = ninstance if j == n_batch-1 else (j+1) * bs
                batch_input_data = {key: input_data[key][start_idx:end_idx] for key in input_data}
                batch_out = estimator(batch_input_data)
                for key in out:
                    out[key].append(batch_out[key])
            for key in out:
                out[key] = torch.cat(out[key], dim=0)
                
            # out = model(input_data)
        
        if 'pred_pose_score' in out.keys():
            pose_scores = out['pred_pose_score'] * out['score']
        else:
            pose_scores = out['score']
        pose_scores = pose_scores.detach().cpu().numpy()
        pred_rot = out['pred_R'].detach().cpu().numpy()
        pred_trans = out['pred_t'].detach().cpu().numpy() * 1000

        print("=> applying Non-Maximum Suppression...")
        
        # First, update all detections with pose estimation results
        scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
        for idx, det in enumerate(detections):
            detections[idx]['score'] = float(pose_scores[idx])
            detections[idx]['R'] = list(pred_rot[idx].tolist())
            detections[idx]['t'] = list(pred_trans[idx].tolist())
            detections[idx]['s'] = list(scale.tolist())
        
        # Apply Non-Maximum Suppression to filter overlapping detections
        print(f"=> Before NMS: {len(detections)} detections")

        # TODO: remove this range(len(detections)) # 
        nms_indices = non_max_suppression(detections, iou_threshold=0.2)
        print(f"=> After NMS: {len(nms_indices)} detections")
        
        # Filter detections, predictions, and other data based on NMS results
        filtered_detections = [detections[i] for i in nms_indices]
        filtered_pred_rot = pred_rot[nms_indices]
        filtered_pred_trans = pred_trans[nms_indices]
        filtered_pose_scores = pose_scores[nms_indices]

        # Save filtered results
        print("=> saving filtered results ...")
        os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)
        with open(os.path.join(f"{args.output_dir}/sam6d_results", 'detection_pem.json'), "w") as f:
            json.dump(filtered_detections, f)
        
        # Update detections for visualization
        detections = filtered_detections

        print("=> visualizating ...")
        
        for idx in range(len(detections)):
            K = np.expand_dims(input_data['K'].detach().cpu().numpy()[idx], axis=0)
            save_path = os.path.join(f"{args.output_dir}/sam6d_results", 'vis_pem_server'+str(idx)+'.png')
            vis_img = visualize_pose_estimation(img, 
                np.expand_dims(filtered_pred_rot[idx], axis=0), 
                np.expand_dims(filtered_pred_trans[idx], axis=0), 
                model_points*1000, K, save_path,
                score=detections[idx]['score']
                )
            vis_img.save(save_path)