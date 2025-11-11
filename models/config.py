gpu_id = '0'
lr = 10 ** (-1.25)
iters = 1500
# 原本是1500
eps = 0.2
# 原本是0.2
beta = 0.5
gamma = 2e-5
psf = 2 # the factor of pixel shuffle

contrast_factor = 0.9
# cover and secret data
cover_dataset_dir = "C:/Users\zja\Desktop\Cs-FNNS-main\Cs-FNNS-main/visual_examples\cover"
cover_image_size = 512
secret_dataset_dir =  "C:/Users\zja\Desktop\Cs-FNNS-main\Cs-FNNS-main/visual_examples\secret"
secret_image_size = 256 # 128 for Cs-FNNS-JPEG 原本为256


# Anti-steganalysis
use_grad_signals_in_steganalysis_nets = True
pre_trained_siastegnet_path = 'steganalysis_networks/siastegnet/checkpoint/odih1/model_best.pth.tar'
pre_trained_srnet_path = 'steganalysis_networks/srnet/checkpoints/net.pt'
pre_trained_yenet_path = 'steganalysis_networks/yenet/checkpoints/net.pt'



# Cs-FNNS-JPEG
attack_layer = 'jpeg'
qf = 80    # quantization factor for jpeg compression
add_jpeg_layer = True
salt_prob = pepper_prob = 0.01
brightness_factor = 1.05
# hiding different images for different users
num_of_receivers = 2
num_of_secret_imgs = 7
gaussian_std = 0.01
contrast_factor = 0.9
save_images = True
resi_magnification = 10
