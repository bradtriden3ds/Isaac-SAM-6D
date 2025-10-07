from omegaconf import OmegaConf

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

