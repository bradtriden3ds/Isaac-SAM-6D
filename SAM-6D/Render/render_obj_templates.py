import blenderproc as bproc

import os
import argparse
import json
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--obj_path', type=str, help="The path of obj model")
parser.add_argument('--output_dir', type=str, help="The path to save CAD templates")
args = parser.parse_args()

print(args.obj_path)
print(args.output_dir)

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')
# cnos_cam_fpath = "/home/yizhou/Projects/SAM-6D/SAM-6D/Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy"

bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)

if not os.path.exists(args.obj_path):
    raise ValueError(f"Object file {args.obj_path} does not exist")

obj_fpath = args.obj_path
scale = get_norm_info(obj_fpath)

# load cnos camera pose
cam_poses = np.load(cnos_cam_fpath)

for idx, cam_pose in enumerate(cam_poses):
    bproc.clean_up()

    # load object
    obj = bproc.loader.load_obj(obj_fpath)[0]
    obj.set_scale([scale, scale, scale])
    obj.set_cp("category_id", 1)

    # convert cnos camera poses to blender camera poses
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
    bproc.camera.add_camera_pose(cam_pose)

    # set light
    light_scale = 2.5
    light_energy = 1000
    light1 = bproc.types.Light()
    light1.set_type("POINT")
    light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    light1.set_energy(light_energy)

    bproc.renderer.set_max_amount_of_samples(50)

    # render the whole pipeline
    data = bproc.renderer.render()
    # render nocs
    data.update(bproc.renderer.render_nocs())

    # check save folder
    save_fpath = os.path.join(args.output_dir, "templates")
    if not os.path.exists(save_fpath):
        os.makedirs(save_fpath)

    # save rgb image
    color_bgr_0 = data["colors"][0]
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
    cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

    # save mask
    mask_0 = data["nocs"][0][..., -1]
    cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)

    # ################################################## PLY Templates ################################################## 

    # bproc.clean_up()

    # # load object
    # obj = bproc.loader.load_obj(args.cad_path)[0]
    # obj.set_scale([scale, scale, scale])
    # obj.set_cp("category_id", 1)

    # bproc.camera.add_camera_pose(cam_pose)

    # # set light
    # light_scale = 2.5
    # light_energy = 1000
    # light1 = bproc.types.Light()
    # light1.set_type("POINT")
    # light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
    # light1.set_energy(light_energy)

    # bproc.renderer.set_max_amount_of_samples(50)

    # # render the whole pipeline
    # data = bproc.renderer.render()
    # # render nocs
    # data.update(bproc.renderer.render_nocs())
    
    # # save nocs
    # xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
    # np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))