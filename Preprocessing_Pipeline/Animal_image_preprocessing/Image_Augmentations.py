import Augmentor

##############################################################################
####   Image Augmentation Block
##############################################################################

# Initialize the pipeline
aug_image_path = Augmentor.Pipeline(r"D:\Behavioral genetics_V1\Metamorph_scans\Echo Images\orientations\True_Orientations_1500_BG\2_classes")
# aug_image_path = Augmentor.Pipeline(r"C:\Users\Pushkar Bansal\Desktop\for_ppt_augmentation")

# Apply augmentations
# aug_image_path.rotate(probability=0.3, max_left_rotation=3, max_right_rotation=3)
aug_image_path.flip_left_right(0.3)
aug_image_path.flip_top_bottom(0.3)
# aug_image_path.random_distortion(probability=0.2, grid_width=4, grid_height=4, magnitude=8)

# Apply random cropping
aug_image_path.crop_random(probability=0.50, percentage_area=0.50)  # less percentage_area: higher cropping

# Brightness augmentation
aug_image_path.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.4)

# Darkness augmentation
# aug_image_path.random_brightness(probability=0.5, min_factor=0.5, max_factor=0.7)

# Saturation augmentation (color)
aug_image_path.random_color(probability=0.5, min_factor=0.7, max_factor=1.3)

# Generate 1000 samples
aug_image_path.sample(5000)
