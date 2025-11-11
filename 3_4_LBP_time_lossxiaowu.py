import numpy as np
import torch
from imageio.v2 import imread
from torch import nn
import random
import os
from PIL import Image
from math import log10, sqrt, ceil
from skimage.transform import resize
import glob
from natsort import natsorted
import lpips
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from models.decodingNetwork256 import  decodingNetwork
from models.network_dncnn import DnCNN
from utils.model import init_weights
from utils.image import calculate_ssim, calculate_psnr, calculate_mae
from utils.logger import logging, logger_info
from utils.dir import mkdirs
from utils.distortion_layers_all import jpeg_compression_layer, img_jpeg_compression, gaussian_noise_layer, add_gaussian_noise,contrast_adjustment_layer,add_contrast_adjustment
import config as c
base_dir = "results"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
secret_dataset = c.secret_dataset_dir.split('/')[-2]
cover_dataset = c.cover_dataset_dir.split('/')[-2]
logger_name = 'RFNNS'
image_save_dirs = os.path.join('base_dir', 'RFNNS', secret_dataset)
mkdirs(image_save_dirs)
logger_info(logger_name, log_path=os.path.join(image_save_dirs, 'result' + '.log'))
logger = logging.getLogger(logger_name)
logger.info('secret dataset: {:s}'.format(secret_dataset))
logger.info('cover dataset: {:s}'.format(cover_dataset))
logger.info('well-trained steganalysis networks for optimization: {:s}'.format('SiaStegNet and SRNet'))
logger.info('use grad signals in steganalysis nets: {}'.format(c.use_grad_signals_in_steganalysis_nets))
logger.info('beta: {:.2f}'.format(c.beta))
logger.info('gamma: {:.5f}'.format(c.gamma))
logger.info('learning rate: {:.3f}'.format(c.lr))
logger.info('epsilon: {:.2f}'.format(c.eps))
logger.info('number of iterations: {}'.format(c.iters))
logger.info('the size of secret image: {}'.format(c.secret_image_size))
logger.info('the size of cover image: {}'.format(c.cover_image_size))
logger.info('Add JPEG layer before the decoding network: {}'.format(c.add_jpeg_layer))

os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def calculate_lpips(img1, img2):
    """
    计算两张图像的LPIPS距离

    参数：
    img1 : numpy.ndarray - 输入图像1 (H x W x C) [0-255]
    img2 : numpy.ndarray - 输入图像2 (H x W x C) [0-255]
    """
    # 初始化LPIPS模型（单例模式）
    if not hasattr(calculate_lpips, 'loss_fn'):
        calculate_lpips.loss_fn = lpips.LPIPS(net='alex').eval()
        if torch.cuda.is_available():
            calculate_lpips.loss_fn = calculate_lpips.loss_fn.cuda()

    # 转换numpy数组为torch张量
    def process_image(img):
        img_t = torch.from_numpy(img).float() / 127.5 - 1.0  # 归一化到[-1,1]
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        if torch.cuda.is_available():
            img_t = img_t.cuda()
        return img_t

    img1_t = process_image(img1)
    img2_t = process_image(img2)

    # 计算LPIPS距离
    with torch.no_grad():
        distance = calculate_lpips.loss_fn(img1_t, img2_t)

    # 转换为标量值
    lpips_value = distance.item()

    # 保持与给定函数一致的异常处理模式
    if lpips_value == 0:
        return float('inf')

    return lpips_value
if c.cover_image_size // c.secret_image_size == 1:
    down_ratio_l3 = 1;
    down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 2:
    down_ratio_l3 = 2;
    down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 4:
    down_ratio_l3 = 2;
    down_ratio_l2 = 2
else:
    print('The code does not take into account the current situation, please adjust the image resulation')

def calculate_lbp_complexity(img_patch):
    """
    计算8x8图像块的LBP纹理复杂度（基于直方图熵）
    输入：img_patch - [8, 8] 灰度张量
    输出：entropy - 纹理复杂度标量值
    """
    patch_size = 8
    lbp = torch.zeros_like(img_patch, dtype=torch.int32)

    # 使用向量化操作加速LBP计算
    # 定义8个邻域位置的偏移量 (dy, dx)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0),
               (1, -1), (0, -1)]

    # 生成邻域比较矩阵
    center = img_patch[1:-1, 1:-1]  # [6,6]
    binary_codes = torch.zeros((8, 6, 6), dtype=torch.bool, device=img_patch.device)

    for k, (dy, dx) in enumerate(offsets):
        neighbor = img_patch[1 + dy:1 + dy + 6, 1 + dx:1 + dx + 6]
        binary_codes[k] = (neighbor > center)

    # 计算LBP编码（使用位运算向量化）
    lbp_values = torch.zeros((6, 6), dtype=torch.int32, device=img_patch.device)
    for k in range(8):
        lbp_values |= binary_codes[k].int() << (7 - k)

    # 填充到原始尺寸（边缘置零）
    lbp[1:-1, 1:-1] = lbp_values

    # 计算直方图熵（忽略边缘零值）
    valid_lbp = lbp[1:-1, 1:-1].flatten().float()  # 只取有效区域的36个值
    hist = torch.histc(valid_lbp, bins=256, min=0, max=255)
    hist = hist[hist > 0] + 1e-10  # 避免除零
    prob = hist / hist.sum()
    entropy = -torch.sum(prob * torch.log2(prob))

    return entropy


def find_nearest_square(n):
    """找到不小于n的最小平方数（至少289）"""
    if n < 289:
        return 289
    m = ceil(sqrt(n))
    return m * m


from steganalysis_networks.siastegnet.val import preprocess_data
from steganalysis_networks.siastegnet.src.models import accuracy, KeNet
from steganalysis_networks.srnet.model.model import Srnet
from steganalysis_networks.yenet.model.YeNet import YeNet


def reverse_crop_and_rearrange_no_loop(pert_full, block_positions, block_size=8):
    N, C, H, W = pert_full.shape
    blocks = (pert_full
              .unfold(2, block_size, block_size)
              .unfold(3, block_size, block_size))
    blocks = blocks.contiguous().view(N, C, -1, block_size, block_size)
    row_col_tensor = torch.tensor(block_positions, device=pert_full.device)  # [num_blocks, 2]
    rows, cols = row_col_tensor[:, 0], row_col_tensor[:, 1]
    block_indices = rows * (W // block_size) + cols  # [num_blocks]

    selected_blocks = blocks[:, :, block_indices, :, :]  # [1, 3, num_blocks, block_size, block_size]

    num_blocks = len(block_positions)
    grid_size = int(np.sqrt(num_blocks))

    selected_blocks = selected_blocks.view(N, C, grid_size, grid_size, block_size, block_size)

    new_pert = (selected_blocks
                .permute(0, 1, 2, 4, 3, 5)
                .contiguous()
                .view(N, C, grid_size*block_size, grid_size*block_size))

    return new_pert



# parparing decoding netwrok, steganalysis networks, and denosing model
model = decodingNetwork(input_channel=3 *c.psf*c.psf, output_channels=3*c.psf*c.psf, down_ratio_l2=down_ratio_l2,
                        down_ratio_l3=down_ratio_l3).to(device)
denoise_model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R').to(device)
denoise_model.load_state_dict(torch.load('models/dncnn_color_blind.pth'), strict=True)
# LpipsNet = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0]) # For calculating LPIPS
SiaStegNet = KeNet().to(device)
SiaStegNet.load_state_dict(torch.load(c.pre_trained_siastegnet_path, map_location='cuda')['state_dict'], strict=False)
SiaStegNet.eval()
SRNet = Srnet().to(device)
SRNet.load_state_dict(torch.load(c.pre_trained_srnet_path)["model_state_dict"])
SRNet.eval()
Yenet = YeNet().to(device)
Yenet.load_state_dict(torch.load(c.pre_trained_yenet_path)["model_state_dict"])
Yenet.eval()

l_rev = torch.nn.MSELoss()
l_hid = torch.nn.MSELoss()
l_JPEG = torch.nn.MSELoss()


l_anti_dec = nn.CrossEntropyLoss()

stego_psnr_list = [];
stego_ssim_list = [];
stego_lpips_list = [0];
stego_apd_list = []
secret_rev_psnr_list = [];
secret_rev_ssim_list = [];
secret_rev_lpips_list = [0];
secret_rev_apd_list = []
YeNet_dec_acc = [0, 0];
SRNet_dec_acc = [0, 0];
SiaStegNet_dec_acc = [0, 0]  # [cover, stego]
secret_image_path_list = list(natsorted(glob.glob(os.path.join(c.secret_dataset_dir, '*'))))
cover_image_path_list = list(natsorted(glob.glob(os.path.join(c.cover_dataset_dir, '*'))))
num_of_imgs = len(secret_image_path_list)
def get_block_view(tensor, block_size):
    return (tensor
            .unfold(2, block_size, block_size)
            .unfold(3, block_size, block_size))
for i in range(len(secret_image_path_list)):

    logger.info('*' * 60)
    logger.info('hiding {}-th image'.format(i))

    # load secret image
    secret = imread(secret_image_path_list[i], pilmode='RGB') / 255.0
    secret = resize(secret, (c.secret_image_size, c.secret_image_size))
    secret = torch.FloatTensor(secret).permute(2, 1, 0).unsqueeze(0).to(device)

    # load cover image
    cover = imread(cover_image_path_list[i], pilmode='RGB') / 255.0
    cover = resize(cover, (c.cover_image_size, c.cover_image_size))
    cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0).to(device)
    cover_backup = cover.clone()

    # praparing decoder
    random_seed_for_decodor = random.randint(0, 100000000)
    logger.info('random_seed_for_decodor(receiver): {:s}'.format(str(random_seed_for_decodor)))
    init_weights(model, random_seed_for_decodor)
    model = model.to(device)
    model.eval()

    block_size = 8
    gray_cover = 0.299 * cover[0, 0] + 0.587 * cover[0, 1] + 0.114 * cover[0, 2]
    patches = gray_cover.unfold(0, 8, 8).unfold(1, 8, 8).contiguous().view(-1, 8, 8)

    # 计算复杂度并排序
    complexity_list = torch.stack([calculate_lbp_complexity(patch) for patch in patches])
    sorted_complexity, sorted_indices = torch.sort(complexity_list, descending=True)

    # 动态阈值
    threshold = 4.5
    valid_mask = (complexity_list > threshold)
    valid_indices = torch.nonzero(valid_mask).squeeze()
    num_valid = valid_indices.size(0)

    # 找到最近的平方数
    target_num = find_nearest_square(num_valid)

    # 调整选中块数量
    if num_valid > target_num:
        selected_indices = sorted_indices[:target_num]
    elif num_valid < target_num:
        pad_num = target_num - num_valid
        selected_indices = torch.cat([sorted_indices[:num_valid], sorted_indices[-pad_num:]])
    else:
        selected_indices = sorted_indices[:target_num]

    # 生成cover_resized
    grid_size = int(sqrt(target_num))
    new_size = grid_size * 8
    cover_resized = torch.zeros((1, 3, new_size, new_size), device=device)
    block_positions = []

    for j, idx in enumerate(selected_indices):
        row = idx.item() // 64
        col = idx.item() % 64
        block_positions.append((row, col))
        block = cover[:, :, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
        y_in_new = (j // grid_size) * 8
        x_in_new = (j % grid_size) * 8
        cover_resized[:, :, y_in_new:y_in_new + 8, x_in_new:x_in_new + 8] = block
    block_positions = []
    for idx in selected_indices:
        row = idx.item() // (cover.shape[3] // 8)  # 原图横向块数
        col = idx.item() % (cover.shape[3] // 8)  # 原图纵向块数
        block_positions.append((row, col))
    # get the lower and upper bound of the perturbation
    mask_pos = torch.gt((torch.ones_like(cover_resized) - cover_resized),
                        (torch.ones_like(cover_resized) * c.eps)).int()
    mask_neg = torch.gt(cover_resized, (torch.ones_like(cover_resized) * c.eps)).int()

    U = (torch.ones_like(cover_resized) * c.eps) * mask_pos + (torch.ones_like(cover_resized) - cover_resized) * (
                1 - mask_pos)
    L = -1 * ((torch.ones_like(cover_resized) * c.eps) * mask_neg + cover_resized * (1 - mask_neg))


    # optimizator
    w_pert = torch.autograd.Variable(torch.zeros_like(cover_resized).float()).to(device)
    w_pert.requires_grad = True
    optimizer = torch.optim.Adam([w_pert], lr=c.lr)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 450, gamma=0.5)

    # 在预处理阶段生成掩码矩阵和块索引
    H, W = cover.shape[2], cover.shape[3]

    # 在迭代循环中替换循环部分
    grid_size = int(np.sqrt(len(block_positions)))

    w_zero_pert = torch.autograd.Variable(torch.zeros_like(w_pert).float()).to(device)

    row_col_tensor = torch.tensor(block_positions, device=device)  # [N, 2]
    rows, cols = row_col_tensor[:, 0], row_col_tensor[:, 1]  # 分别是 row 和 col

    adv_pert_full = torch.zeros_like(cover, dtype=cover.dtype, device=device)

    block_indices = rows * (W // block_size) + cols  # [N]

    cover_blocks = get_block_view(adv_pert_full, block_size)  # 同理
    cover_shape = cover_blocks.shape  # 形状: [1, 3, unfold_H, unfold_W, block_size, block_size]
    cover_blocks = cover_blocks.contiguous().view(1, 3, -1, block_size, block_size)

    for iteration_index in range(c.iters):
        # print(iteration_index)
        optimizer.zero_grad()
        adv_pert = L + (U - L) * ((torch.tanh(w_pert) + 1) / 2)


        adv_blocks = get_block_view(adv_pert, block_size)  # [1, 3, unfold_H, unfold_W, block_size, block_size]
        adv_blocks = adv_blocks.contiguous().view(1, 3, -1, block_size, block_size)


        cover_blocks[:, :, block_indices] = adv_blocks[:, :, :len(block_indices)]


        adv_pert_full = cover_blocks.view(cover_shape)  # [1, 3, unfold_H, unfold_W, block_size, block_size]
        adv_pert_full = adv_pert_full.permute(0, 1, 2, 4, 3, 5).contiguous()  # 交换维度
        adv_pert_full = adv_pert_full.view(1, 3, H, W)  # 最终得到 [1, 3, H, W] 的完整扰动图


        # adv_pert_full_JPEG = contrast_adjustment_layer(adv_pert_full,cover)

        adv_pert = reverse_crop_and_rearrange_no_loop(adv_pert_full, block_positions)

        # adv_pert_JPEG = reverse_crop_and_rearrange_no_loop(adv_pert_full_JPEG, block_positions)

        output = model(adv_pert)
        # output_JPEG = model(adv_pert_JPEG)
        loss_1 = l_hid(adv_pert, w_zero_pert)
        loss_2 = l_rev(output, secret)
        # loss_3 = l_JPEG(output_JPEG, secret)

        if loss_1.item() < 0.001:
            loss = 0.001 + loss_2 * 3
        else:
            # 优先优化隐藏损失直到达标
            loss = loss_1 + loss_2 * 3
        if c.use_grad_signals_in_steganalysis_nets and (iteration_index + 1) > 1400:
            # SRNet
            adv_pert_full = torch.zeros_like(cover)
            grid_size = int(np.sqrt(len(block_positions)))  # 根据选中块数量确定网格尺寸
            block_size = 8

            # 遍历所有选中的块位置
            for j, (row, col) in enumerate(block_positions):
                # 计算在adv_pert中的块位置
                y_in_pert = (j // grid_size) * block_size
                x_in_pert = (j % grid_size) * block_size

                # 从adv_pert提取块
                block = adv_pert[:, :, y_in_pert:y_in_pert + block_size, x_in_pert:x_in_pert + block_size]

                # 将块放回原图对应位置
                adv_pert_full[:, :, row * block_size:(row + 1) * block_size,
                col * block_size:(col + 1) * block_size] = block
            inputs = cover_backup + adv_pert_full
            labels = torch.tensor([0]).to(device)
            outputs = SRNet(inputs)
            loss += c.beta * c.gamma * l_anti_dec(outputs, labels)
            # # siastegnet
            inputs, labels = preprocess_data((cover_backup + adv_pert_full) * 255, torch.tensor([0]).to(device), False)
            outputs, feats_0, feats_1 = SiaStegNet(*inputs)
            loss += c.beta * c.gamma * l_anti_dec(outputs, labels)

        loss.backward(retain_graph=True)
        optimizer.step()
        weight_scheduler.step()

    logger.info('-' * 60)
    # rounding and clipping operations
    adv_image = cover + adv_pert_full
    adv_image = torch.round(torch.clamp(adv_image * 255, min=0., max=255.)) / 255
    # adv_pert_full = add_contrast_adjustment(adv_image) - cover
    adv_pert_full = adv_image - cover
    # testing
    adv_pert = reverse_crop_and_rearrange_no_loop(adv_pert_full, block_positions)
    secret_rev = model(adv_pert)

    # anti-steganalysis
    # siastegnet
    inputs, labels = preprocess_data(cover * 255, torch.tensor([0.]), False)
    outputs, feats_0, feats_1 = SiaStegNet(*inputs)
    SiaStegNet_dec_acc[0] += accuracy(outputs, labels).item()
    inputs, labels = preprocess_data(adv_image * 255, torch.tensor([1.]), False)
    outputs, feats_0, feats_1 = SiaStegNet(*inputs)
    SiaStegNet_dec_acc[1] += accuracy(outputs, labels).item()
    logger.info('SiaStegNet_dec_acc: {}'.format(SiaStegNet_dec_acc))

    # # srnet
    inputs = cover;labels = torch.tensor([0.]).to(device)
    outputs = SRNet(inputs)
    SRNet_dec_acc[0] += accuracy(outputs, labels).item()
    inputs = adv_image;labels = torch.tensor([1.]).to(device)
    outputs = SRNet(inputs)
    SRNet_dec_acc[1] += accuracy(outputs, labels).item()
    logger.info('SRNet_dec_acc: {}'.format(SRNet_dec_acc))

    # yenet
    inputs = cover;
    labels = torch.tensor([0.]).to(device)
    outputs = Yenet(inputs)
    YeNet_dec_acc[0] += accuracy(outputs, labels).item()
    inputs = adv_image;
    labels = torch.tensor([1.]).to(device)
    outputs = Yenet(inputs)
    YeNet_dec_acc[1] += accuracy(outputs, labels).item()
    logger.info('YeNet_dec_acc: {}'.format(YeNet_dec_acc))

    # denosing the recovered secret images
    secret_rev = denoise_model(secret_rev)
    secret_rev = torch.round(torch.clamp(secret_rev * 255, min=0., max=255.)) / 255
    cover_resi = (adv_image - cover).abs() * c.resi_magnification
    secret_resi = (secret_rev - secret).abs() * c.resi_magnification

    # tensor(cuda) to numpy(cpu)
    cover = cover.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    stego = adv_image.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    secret = secret.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    secret_rev = secret_rev.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    cover_resi = cover_resi.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    secret_resi = secret_resi.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255

    # calculing and recoding SSIM and PSNR
    stego_psnr = calculate_psnr(cover, stego)
    stego_ssim = calculate_ssim(cover, stego)

    stego_lpips = calculate_lpips(cover, stego)

    secret_rev_psnr = calculate_psnr(secret, secret_rev)
    secret_rev_ssim = calculate_ssim(secret, secret_rev)

    secret_rev_lpips = calculate_lpips(secret, secret_rev)

    stego_apd = calculate_mae(cover, stego)
    secret_rev_apd = calculate_mae(secret, secret_rev)
    logger.info('stego_psnr: {:.2f}, secret_rev_psnr: {:.2f}'.format(stego_psnr, secret_rev_psnr))
    logger.info('stego_ssim: {:.4f}, secret_rev_ssim: {:.4f}'.format(stego_ssim, secret_rev_ssim))

    logger.info('stego_lpips: {:.4f}, secret_rev_lpips: {:.4f}'.format(stego_lpips, secret_rev_lpips))

    logger.info('stego_apd: {:.2f}, secret_rev_apd: {:.2f}'.format(stego_apd, secret_rev_apd))
    stego_psnr_list.append(stego_psnr)
    secret_rev_psnr_list.append(secret_rev_psnr)
    stego_ssim_list.append(stego_ssim)
    secret_rev_ssim_list.append(secret_rev_ssim)
    stego_apd_list.append(stego_apd)
    secret_rev_apd_list.append(secret_rev_apd)

    stego_lpips_list.append(stego_lpips)
    secret_rev_lpips_list.append(secret_rev_lpips)


    if c.save_images:
        cover_save_path = os.path.join(image_save_dirs, 'cover_jpegloss', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        stego_save_path = os.path.join(image_save_dirs, 'stego_jpegloss', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_save_path = os.path.join(image_save_dirs, 'secret_jpegloss', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_rev_save_path = os.path.join(image_save_dirs, 'secret_rev_jpegloss', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        cover_resi_save_path = os.path.join(image_save_dirs, 'cover_resi_jpegloss', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_resi_save_path = os.path.join(image_save_dirs, 'secret_resi_jpegloss', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        mkdirs(os.path.join(image_save_dirs, 'cover_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'stego_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'secret_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'secret_rev_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'cover_resi_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'secret_resi_jpegloss'))
        logger.info('saving images...')
        Image.fromarray(cover.astype(np.uint8)).save(cover_save_path)
        Image.fromarray(stego.astype(np.uint8)).save(stego_save_path)
        Image.fromarray(secret.astype(np.uint8)).save(secret_save_path)
        Image.fromarray(secret_rev.astype(np.uint8)).save(secret_rev_save_path)
        Image.fromarray(cover_resi.astype(np.uint8)).save(cover_resi_save_path)
        Image.fromarray(secret_resi.astype(np.uint8)).save(secret_resi_save_path)


logger.info('stego_psnr_mean: {:.2f}, stego_ssim_mean: {:.4f}, stego_lpips_mean: {:.4f}, stego_apd_mean: {:.2f}'.format(np.array(stego_psnr_list).mean(), np.array(stego_ssim_list).mean(), np.array(stego_lpips_list).mean(), np.array(stego_apd_list).mean()))
logger.info('secret_rev_psnr_mean: {:.2f}, secret_rev_ssim_mean: {:.4f}, secret_rev_lpips_mean: {:.4f}, secret_rev_apd_mean: {:.2f}'.format(np.array(secret_rev_psnr_list).mean(), np.array(secret_rev_ssim_list).mean(), np.array(secret_rev_lpips_list).mean(), np.array(secret_rev_apd_list).mean()))
logger.info('YeNet_cover_det_acc: {:.2f}, YeNet_stego_det_acc: {:.2f}, YeNet_total_det_acc: {:.2f}'.format(YeNet_dec_acc[0]/num_of_imgs ,YeNet_dec_acc[1]/num_of_imgs, (YeNet_dec_acc[0]+YeNet_dec_acc[1])/2/num_of_imgs))
logger.info('SRNet_cover_det_acc: {:.2f}, SRNet_stego_det_acc: {:.2f}, SRNet_total_det_acc: {:.2f}'.format(SRNet_dec_acc[0]/num_of_imgs ,SRNet_dec_acc[1]/num_of_imgs, (SRNet_dec_acc[0]+SRNet_dec_acc[1])/2/num_of_imgs))
logger.info('SiaStegNet_cover_det_acc: {:.2f}, SiaStegNet_stego_det_acc: {:.2f}, SiaStegNet_total_det_acc: {:.2f}'.format(SiaStegNet_dec_acc[0]/num_of_imgs ,SiaStegNet_dec_acc[1]/num_of_imgs, (SiaStegNet_dec_acc[0]+SiaStegNet_dec_acc[1])/2/num_of_imgs))


