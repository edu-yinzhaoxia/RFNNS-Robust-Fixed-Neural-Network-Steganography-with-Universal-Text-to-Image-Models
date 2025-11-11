import torch
import config as c
import torchvision
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
def gaussian_noise_layer(adv_pert):
    return  adv_pert + torch.randn(adv_pert.shape).mul_(c.sigma/255).to(adv_pert.device)

def poisson_noise_layer(adv_pert):
    return  adv_pert + torch.poisson(torch.rand(adv_pert.shape)).mul_(c.sigma/255).to(adv_pert.device)

transform_to_pil = torchvision.transforms.ToPILImage()
transform_to_tensor = torchvision.transforms.ToTensor()
ps = torch.nn.PixelShuffle(c.psf)
pus = torch.nn.PixelUnshuffle(c.psf)

def jpeg_compression_layer(adv_pert, cover):
    adv_image = cover + adv_pert
    adv_image = adv_image.squeeze(dim=0).cpu()
    adv_image = transform_to_pil(adv_image)
    outputIoStream = BytesIO()
    adv_image.save(outputIoStream, "JPEG", quality=c.qf)
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(adv_pert.device)
    jpeg_noise = (adv_image_jpeg - (cover + adv_pert)).detach()
    return adv_pert + jpeg_noise

def gaussian_noise_layer(adv_pert, cover):
    device = adv_pert.device
    adv_image = cover + adv_pert
    noise = torch.randn_like(adv_image) * c.gaussian_std
    noisy_image = torch.clamp(adv_image + noise, 0.0, 1.0)
    noise_residual = (noisy_image - adv_image).detach()
    noisy_pert = adv_pert + noise_residual
    return noisy_pert

def contrast_adjustment_layer(adv_pert, cover):
    """
    调整对比度的函数
    输入：
        adv_pert - 对抗扰动张量 [batch_size, channels, height, width]
        cover - 原始图像张量 [batch_size, channels, height, width]
    输出：
        contrast_pert - 调整对比度后的对抗扰动张量
    """
    device = adv_pert.device
    adv_image = cover + adv_pert

    # 调整对比度
    contrast_factor = c.contrast_factor  # 从config中获取对比度调整参数
    mean = torch.mean(adv_image, dim=(2, 3), keepdim=True)
    contrast_image = mean + contrast_factor * (adv_image - mean)

    # 截断到合法范围
    contrast_image = torch.clamp(contrast_image, 0.0, 1.0)

    # 计算对比度调整残差
    contrast_residual = (contrast_image - adv_image).detach()

    # 保持原始梯度路径
    contrast_pert = adv_pert + contrast_residual

    return contrast_pert

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    LL = x1 + x2 + x3 + x4
    LH = -x1 - x2 + x3 + x4
    HL = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, LH, HL, HH

def idwt_init(LL, LH, HL, HH):
    batch, channel, h, w = LL.shape
    x = torch.zeros((batch, channel, h*2, w*2), device=LL.device)
    x[:, :, 0::2, 0::2] = (LL - LH - HL + HH) / 2
    x[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / 2
    x[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / 2
    x[:, :, 1::2, 1::2] = (LL + LH + HL + HH) / 2
    return x


def rotation_attack_layer(adv_pert, cover, max_angle=0.1):
    """
    旋转攻击层 - 用于对抗训练流程
    输入:
        adv_pert - 对抗扰动张量 [batch_size, channels, height, width]
        cover - 原始图像张量 [batch_size, channels, height, width]
        max_angle - 最大旋转角度(度), 控制攻击强度
    输出:
        rotated_pert - 旋转后的对抗扰动张量
    """
    device = adv_pert.device
    adv_image = cover + adv_pert
    batch_size = adv_image.shape[0]

    # 生成随机旋转角度 (-max_angle 到 max_angle 之间)
    angles = (torch.rand(batch_size) * 2 * max_angle - max_angle)  # 这里补上了缺失的右括号

    # 对每张图像分别进行旋转
    rotated_image = torch.zeros_like(adv_image)
    for i in range(batch_size):
        # 转换为PIL图像进行旋转
        img_pil = transform_to_pil(adv_image[i].cpu())
        img_pil = img_pil.rotate(angles[i].item(), resample=Image.BILINEAR, expand=False)
        rotated_image[i] = transform_to_tensor(img_pil).to(device)

    # 计算旋转残差并保持梯度
    rotation_residual = (rotated_image - adv_image).detach()
    rotated_pert = adv_pert + rotation_residual

    return rotated_pert

def scaling_attack_layer(adv_pert, cover, scale_factor=0.95):
    """
    缩放攻击层 - 用于对抗训练流程
    输入:
        adv_pert - 对抗扰动张量 [batch_size, channels, height, width]
        cover - 原始图像张量 [batch_size, channels, height, width]
        scale_factor - 缩放因子 (0-1之间), 控制攻击强度
    输出:
        scaled_pert - 缩放后的对抗扰动张量
    """
    device = adv_pert.device
    adv_image = cover + adv_pert
    batch_size, _, h, w = adv_image.shape

    # 计算缩放后的尺寸
    scaled_h = int(h * scale_factor)
    scaled_w = int(w * scale_factor)

    # 先缩小图像
    scaled_down = F.interpolate(adv_image, size=(scaled_h, scaled_w),
                                mode='bilinear', align_corners=False)

    # 再放大回原尺寸
    scaled_image = F.interpolate(scaled_down, size=(h, w),
                                 mode='bilinear', align_corners=False)

    # 计算缩放残差并保持梯度
    scale_residual = (scaled_image - adv_image).detach()
    scaled_pert = adv_pert + scale_residual

    return scaled_pert

def jpeg_compression_dwt_layer(cover, LL_adv_pert):
    LL, LH, HL, HH = dwt_init(cover)
    LL_new = LL + LL_adv_pert
    adv_image = idwt_init(LL_new, LH, HL, HH)
    adv_image = adv_image.squeeze(dim=0).cpu()
    adv_image = transform_to_pil(adv_image)
    outputIoStream = BytesIO()
    adv_image.save(outputIoStream, "JPEG", quality=c.qf)
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(cover.device)
    LL_JPEG, LH_JPEG, HL_JPEG, HH_JPEG = dwt_init(adv_image_jpeg)
    jpeg_noise = (LL_JPEG - LL_new).detach()
    return LL_adv_pert + jpeg_noise

def attack_layer(adv_pert, cover):
    if c.attack_layer == 'gaussian':
        return gaussian_noise_layer(adv_pert)
    elif c.attack_layer == 'possion':
        return poisson_noise_layer(adv_pert)
    elif c.attack_layer == 'contrast':
        return contrast_adjustment_layer(adv_pert, cover)
    elif c.attack_layer == 'scale':
        return scaling_attack_layer(adv_pert, cover, c.scale_factor)
    elif c.attack_layer == 'rotate':
        return rotation_attack_layer(adv_pert, cover, c.max_angle)
    else:  # jpeg
        return jpeg_compression_layer(adv_pert, cover)

def img_jpeg_compression(adv_image):
    device = adv_image.device
    adv_image = transform_to_pil(adv_image.squeeze(dim=0).cpu())
    outputIoStream = BytesIO()
    adv_image.save(outputIoStream, "JPEG", quality=c.qf)
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(device)
    return adv_image_jpeg

def add_gaussian_noise(adv_image):
    device = adv_image.device
    batch_size = adv_image.shape[0]
    noise = torch.randn_like(adv_image) * c.gaussian_std
    noised_image = torch.clamp(adv_image + noise, 0.0, 1.0)
    if adv_image.requires_grad:
        noised_image = noised_image - adv_image.detach() + adv_image
    return noised_image

def add_contrast_adjustment(adv_image):
    """
    添加对比度调整到对抗扰动图像
    输入：
        adv_image - 张量 [batch_size, channels, height, width]，值范围[0,1]
    输出：
        contrast_image - 调整对比度后的张量
    """
    device = adv_image.device
    mean = torch.mean(adv_image, dim=(2, 3), keepdim=True)
    contrast_factor = c.contrast_factor  # 从config中获取对比度调整参数
    contrast_image = mean + contrast_factor * (adv_image - mean)
    contrast_image = torch.clamp(contrast_image, 0.0, 1.0)
    if adv_image.requires_grad:
        contrast_image = contrast_image - adv_image.detach() + adv_image
    return contrast_image


def add_scaling_attack(adv_image, scale_factor=0.95):
    """
    添加缩放攻击到对抗图像
    输入:
        adv_image - 张量 [batch_size, channels, height, width]
        scale_factor - 缩放因子 (0-1之间), 控制攻击强度
    输出:
        scaled_image - 缩放后的图像
    """
    device = adv_image.device
    batch_size, _, h, w = adv_image.shape

    # 计算缩放后的尺寸
    scaled_h = int(h * scale_factor)
    scaled_w = int(w * scale_factor)

    # 先缩小图像
    scaled_down = F.interpolate(adv_image, size=(scaled_h, scaled_w),
                                mode='bilinear', align_corners=False)

    # 再放大回原尺寸
    scaled_image = F.interpolate(scaled_down, size=(h, w),
                                 mode='bilinear', align_corners=False)

    # 保持梯度信息
    if adv_image.requires_grad:
        scaled_image = scaled_image - adv_image.detach() + adv_image

    return scaled_image


def add_rotation_attack(adv_image, max_angle=0.1):
    """
    添加旋转攻击到对抗图像
    输入:
        adv_image - 张量 [batch_size, channels, height, width]
        max_angle - 最大旋转角度(度), 控制攻击强度
    输出:
        rotated_image - 旋转后的图像
    """
    device = adv_image.device
    batch_size = adv_image.shape[0]

    # 生成随机旋转角度 (-max_angle 到 max_angle 之间)
    angles = (torch.rand(batch_size) * 2 * max_angle - max_angle).to(device)

    # 对每张图像分别进行旋转
    rotated_image = torch.zeros_like(adv_image)
    for i in range(batch_size):
        # 转换为PIL图像进行旋转
        img_pil = transform_to_pil(adv_image[i].cpu())
        img_pil = img_pil.rotate(angles[i].item(), resample=Image.BILINEAR, expand=False)
        rotated_image[i] = transform_to_tensor(img_pil).to(device)

    # 保持梯度信息
    if adv_image.requires_grad:
        rotated_image = rotated_image - adv_image.detach() + adv_image

    return rotated_image

