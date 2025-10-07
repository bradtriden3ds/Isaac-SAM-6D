export SAM_6D_FOLDER=/home/uxsimdeu/workspaces/Isaac-SAM-6D/SAM-6D
export SEGMENTOR_MODEL=sam
export datadir=/mnt/e/users/uqr/IsaacSim/2025_10_05_SAM6D_Calibration_BoxOrigin
export OUTPUT_DIR=$datadir/output/BOX_OBJ
export OBJ_PATH=$datadir/model/BOX_OBJ.obj 
export CAD_PATH=$datadir/model/BOX_OBJ.ply
export RGB_PATH=$datadir/images/isaacsim_camera_capture_20_left.png
export DEPTH_PATH=$datadir/depth/depth_map.png
export CAMERA_PATH=$datadir/camerainfo/camera_1280x720.json

# check if /templates exists under OUTPUT_DIR, if not run the following blenderproc command to generate it
if [ ! -d "$OUTPUT_DIR/templates" ]; then
    echo "Templates folder does not exist. Generating templates..."
    blenderproc run ./Render/render_obj_templates.py --output_dir $OUTPUT_DIR --obj_path $OBJ_PATH --ply_path $CAD_PATH
fi

python start_server.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --segmentor_model $SEGMENTOR_MODEL