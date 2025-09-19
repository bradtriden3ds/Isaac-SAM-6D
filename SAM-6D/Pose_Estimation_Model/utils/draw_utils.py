import numpy as np
import os
import cv2
import pycocotools.mask as cocomask

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_detections(image, pred_rots, pred_trans, model_points, intrinsics, color=(255, 0, 0)):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    # 3d bbox
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    # 3d point
    choose = np.random.choice(np.arange(len(model_points)), 512)
    pts_3d = model_points[choose].T

    for ind in range(num_pred_instances):
        # draw 3d bounding box
        transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
        # draw point cloud
        transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

    return draw_image_bbox

def draw_text(img, text, position, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    # Convert text to string if it's not already
    if isinstance(text, (int, float)):
        text = f"{text:.3f}"
    elif hasattr(text, 'item'):  # Handle numpy scalars or tensors
        text = f"{text.item():.3f}"
    else:
        text = str(text)
    cv2.putText(img, text, position, font, font_scale, color, thickness)
    return img


def bbox_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1, bbox2: [x,y, w, h]
    
    Returns:
        iou: float, IoU value between 0 and 1
    """
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1
    x2_1, y2_1, w2, h2 = bbox2
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2
    intersection = max(0, min(x1_2, x2_2) - max(x1_1, x2_1)) * max(0, min(y1_2, y2_2) - max(y1_1, y2_1))
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def mask_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1, mask2: binary numpy arrays
    
    Returns:
        iou: float, IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0


def segmentation_iou(seg1, seg2, image_size):
    """
    Calculate IoU between two COCO-style segmentations.
    
    Args:
        seg1, seg2: COCO segmentation format
        image_size: (height, width) tuple
    
    Returns:
        iou: float, IoU value between 0 and 1
    """
    h, w = image_size
    
    # Decode segmentations to masks
    try:
        rle1 = cocomask.frPyObjects(seg1, h, w) if not isinstance(seg1, dict) else seg1
        rle2 = cocomask.frPyObjects(seg2, h, w) if not isinstance(seg2, dict) else seg2
    except:
        rle1 = seg1
        rle2 = seg2
    
    mask1 = cocomask.decode(rle1) > 0
    mask2 = cocomask.decode(rle2) > 0
    
    return mask_iou(mask1, mask2)


def non_max_suppression(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        detections: list of detection dictionaries with 'score', 'bbox' (optional), 'segmentation' (optional)
        iou_threshold: float, IoU threshold for considering detections as overlapping
    
    Returns:
        keep_indices: list of indices to keep after NMS
    """
    if len(detections) == 0:
        return []
    
    # Extract scores
    scores = [det['score'] for det in detections]
    
    # Sort by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Keep the detection with highest score
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
            
        current_det = detections[current_idx]
        remaining_indices = sorted_indices[1:]
        
        # Calculate IoU with remaining detections
        overlaps = []
        for idx in remaining_indices:
            other_det = detections[idx]
            
            if 'bbox' in current_det and 'bbox' in other_det:
                # Use bbox IoU as fallback
                iou = bbox_iou(current_det['bbox'], other_det['bbox'])
                print(f"cbbox IoU: {iou}", "current_det['bbox']", current_det['bbox'], 
                "other_det['bbox']", other_det['bbox'])
            else:
                # If no bbox or mask info, assume no overlap
                iou = 0.0
            
            overlaps.append(iou)
        
        # Remove detections that overlap too much
        overlaps = np.array(overlaps)
        non_overlapping = overlaps <= iou_threshold
        sorted_indices = remaining_indices[non_overlapping]
    
    return keep_indices