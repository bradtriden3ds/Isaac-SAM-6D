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

from Pose_Estimation_Model.model.pose_estimation_model import Net as Estimator
from Pose_Estimation_Model.utils.data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)
from Pose_Estimation_Model.utils.draw_utils import draw_detections, draw_text, non_max_suppression
import pycocotools.mask as cocomask
import trimesh

rgb_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

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

pose_estimation_config = OmegaConf.create({'coarse_npoint': 196, 'fine_npoint': 2048, 'feature_extraction': {'vit_type': 'vit_base', 'up_type': 'linear', 'embed_dim': 768, 'out_dim': 256, 'use_pyramid_feat': True, 'pretrained': True}, 'geo_embedding': {'sigma_d': 0.2, 'sigma_a': 15, 'angle_k': 3, 'reduction_a': 'max', 'hidden_dim': 256}, 'coarse_point_matching': {'nblock': 3, 'input_dim': 256, 'hidden_dim': 256, 'out_dim': 256, 'temp': 0.1, 'sim_type': 'cosine', 'normalize_feat': True, 'loss_dis_thres': 0.15, 'nproposal1': 6000, 'nproposal2': 300}, 'fine_point_matching': {'nblock': 3, 'input_dim': 256, 'hidden_dim': 256, 'out_dim': 256, 'pe_radius1': 0.1, 'pe_radius2': 0.2, 'focusing_factor': 3, 'temp': 0.1, 'sim_type': 'cosine', 'normalize_feat': True, 'loss_dis_thres': 0.15}})

pose_estimation_test_config = OmegaConf.create({'name': 'bop_test_dataset', 'data_dir': '../Data/BOP', 'template_dir': '../Data/BOP-Templates', 'img_size': 224, 'n_sample_observed_point': 2048, 'n_sample_model_point': 1024, 'n_sample_template_point': 5000, 'minimum_n_point': 8, 'rgb_mask_flag': True, 'seg_filter_score': 0.25, 'n_template_view': 42})


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

def visualize_image_segmentation(rgb, detections, save_path="tmp.png"):
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

def visualize_pose_estimation(rgb, pred_rot, pred_trans, model_points, K, save_path, score=None):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    if score is not None:
        img = draw_text(img, score, (10, 60), color=(0, 150, 200))
        # also draw pret_rot and pred_trans 
        img = draw_text(img, "rot: "+ str(np.round(pred_rot.tolist(), 2)), (10, 80), color=(200, 150, 0))
        img = draw_text(img, "trans: "+ str(np.round(pred_trans.tolist(), 2)), (10, 100), color=(0, 200, 0))


    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
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

def _get_template(path, cfg, tem_index=1):
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    rgb = load_im(rgb_path).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  
    mask = load_im(mask_path).astype(np.uint8) == 255

    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb_transform(np.array(rgb))

    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)
    return rgb, rgb_choose, xyz

def get_templates(path, cfg):
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(total_nView):
        i = int(n_template_view / total_nView * v)
        tem, tem_choose, tem_pts = _get_template(path, cfg, i)
        all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
        all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
        all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())
    return all_tem, all_tem_pts, all_tem_choose

def get_test_data(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg):
    dets = []
    with open(seg_path) as f:
        dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation
    for det in dets_:
        if det['score'] > det_score_thresh:
            dets.append(det)
    del dets_

    cam_info = json.load(open(cam_path))
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)

    # if whole_image is RGBA, convert to RGB
    if whole_image.shape[-1] == 4:
        whole_image = whole_image[:,:,:3]

    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info['depth_scale'] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))


    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(torch.FloatTensor(rgb))
        all_cloud.append(torch.FloatTensor(cloud))
        all_rgb_choose.append(torch.IntTensor(rgb_choose).long())
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    ret_dict['pts'] = torch.stack(all_cloud).cuda()
    ret_dict['rgb'] = torch.stack(all_rgb).cuda()
    ret_dict['rgb_choose'] = torch.stack(all_rgb_choose).cuda()
    ret_dict['score'] = torch.FloatTensor(all_score).cuda()

    ninstance = ret_dict['pts'].size(0)
    ret_dict['model'] = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    ret_dict['K'] = torch.FloatTensor(K).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
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

        detector.ref_data = {}
        detector.ref_data["descriptors"] = detector.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        detector.ref_data["appe_descriptors"] = detector.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data

        mesh = trimesh.load_mesh(args.cad_path)
        model_points = mesh.sample(2048).astype(np.float32) / 1000.0
        detector.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)

        # compute the geometric score
        depth_path = args.depth_path
        cam_path = args.cam_path
        batch = batch_input_data(depth_path, cam_path, device)
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(device)
        detector.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

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
        print("=> loading model ...")
        estimator = Estimator(pose_estimation_config).to(device)
        estimator.eval()
        checkpoint = os.path.join("./Pose_Estimation_Model", 'checkpoints', 'sam-6d-pem-base.pth')
        print("=> loading checkpoint ...", checkpoint)
        gorilla.solver.load_checkpoint(model=estimator, filename=checkpoint)

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