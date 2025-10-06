export SAM_6D_FOLDER=/home/yizhou/Projects/SAM-6D/SAM-6D
export SEGMENTOR_MODEL=sam
export OUTPUT_DIR=$SAM_6D_FOLDER/Data/Example6/outputs  
export OBJ_PATH=$SAM_6D_FOLDER/Data/Example6/BOX_OBJ.obj 
export CAD_PATH=$SAM_6D_FOLDER/Data/Example6/BOX_NEW.ply  

# check if /templates exists under OUTPUT_DIR, if not run the following blenderproc command to generate it
if [ ! -d "$OUTPUT_DIR/templates" ]; then
    echo "Templates folder does not exist. Generating templates..."
    blenderproc run ./Render/render_obj_templates.py --output_dir $OUTPUT_DIR --obj_path $OBJ_PATH --ply_path $CAD_PATH
fi

python start_server.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --segmentor_model $SEGMENTOR_MODEL