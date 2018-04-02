from u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

img_rows = 512
img_cols = 512
desired_size = 2570 # Size of square image that we want to make from original data
key_points_desired_size = (2080, 1600)
key_points_display_size = (1040, 800)
key_points_max_size = (4160, 3200)
augmented_sample_amount = 20
color_red = "#ff0000"
color_blue = "#0000ff"
color_yellow = "#ffff00"
unet_test_path = '../mask/unet-test'
unet_train_path = '../mask/unet-train/train'
unet_mask_path = '../mask/unet-mask/train'
unet_train_val_path = '../mask/unet-train/validation'
unet_mask_val_path = '../mask/unet-mask/validation'
mask_raw_path = '../mask/mask-raw'
mask_good_raw_path = '../mask/mask-good-raw'
mask_final_path = '../mask/mask-final'
key_points_in_path = '../key-points/train'
augmented_train_path = '../mask/mask-aug'
# augmented_mask_path = '../mask/mask-aug'

def get_unet():
	return get_unet_512(input_shape=(img_rows, img_cols, 1))