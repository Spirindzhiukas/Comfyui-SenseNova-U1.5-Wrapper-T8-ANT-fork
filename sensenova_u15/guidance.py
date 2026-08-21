def edit_guidance(positive, image_condition, negative, cfg, img_cfg):
    return negative + cfg * (positive - image_condition) + img_cfg * (image_condition - negative)
