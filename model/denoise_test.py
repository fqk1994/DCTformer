import numpy as np
import matplotlib.pyplot as plt
from model import DCTformer
import torch
import torch.nn as nn
from utils.GetPatches import read_segy_data ,create_segy
from utils.Cut_combine import cut, combine,combine_gaussian
import segyio
import os

# 加载数据
def load_seismic_data(file_path):
    print(f"Loading seismic data from {file_path}...")
    seismic_noise = read_segy_data(file_path)
    seismic_block_h, seismic_block_w = seismic_noise.shape
    print(f"Loaded seismic data shape: {seismic_noise.shape}")
    return seismic_noise, seismic_block_h, seismic_block_w

# 数据归一化处理
def normalize_data(data):
    seismic_noise_max = abs(data).max()
    normalized_data = data / seismic_noise_max
    return normalized_data, seismic_noise_max

# 数据填充
def pad_data(data, patch_size):
    seismic_block_h, seismic_block_w = data.shape
    pad_h = patch_size - (seismic_block_h % patch_size) if seismic_block_h % patch_size != 0 else 0
    pad_w = patch_size - (seismic_block_w % patch_size) if seismic_block_w % patch_size != 0 else 0
    padded_data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='reflect')
    print(f"Padded data shape: {padded_data.shape}")
    return padded_data, pad_h, pad_w

# 数据切分
def cut_patches(data, patch_size, stride_x, stride_y):
    print(f"Cutting data into patches with size {patch_size}x{patch_size} and stride {stride_x}x{stride_y}...")
    patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(data, patch_size, stride_x, stride_y)
    return patches, strides_x, strides_y, fill_arr_h, fill_arr_w

# 加载模型
def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = DCTformer()
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model.state_dict()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device=device)
    model.eval()
    return model

# 去噪处理
def denoise_patches(patches, model, device, batch_size):
    predict_datas = []
    print(f"Processing patches in batches of size {batch_size}...")
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_patches_np = np.array([patch for patch in batch_patches])
            batch_patches_tensor = torch.from_numpy(batch_patches_np).reshape(-1, 1, patch_size, patch_size).to(device=device, dtype=torch.float32)
            predict_data_batch = model(batch_patches_tensor)
            predict_datas.extend(predict_data_batch.data.cpu().numpy().squeeze())
    return predict_datas

# 数据还原
def combine_patches(predict_datas, patch_size, strides_x, strides_y, padded_height, padded_width, stride_x, stride_y):
    print("Combining patches back into a single image...")
    seismic_predict_padded = combine_gaussian(predict_datas, patch_size, strides_x, strides_y, padded_height, padded_width, stride_x, stride_y)
    return seismic_predict_padded

# 数据裁剪
def crop_data(data, original_height, original_width):
    print(f"Cropping data to original size: {original_height}x{original_width}...")
    return data[:original_height, :original_width]

# 数据逆归一化
def unnormalize_data(data, max_value):
    return data * max_value

# 可视化结果
def visualize_results(original, denoised, residual):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 3, 1)
    ax2 = fig1.add_subplot(1, 3, 2)
    ax3 = fig1.add_subplot(1, 3, 3)
    ax1.imshow(original, cmap='gray', interpolation='nearest', aspect=0.2, vmin=-0.5, vmax=0.5)
    ax2.imshow(denoised, cmap='gray', interpolation='nearest', aspect=0.2, vmin=-0.5, vmax=0.5)
    ax3.imshow(residual, cmap='gray', interpolation='nearest', aspect=0.2, vmin=-0.5, vmax=0.5)
    plt.tight_layout()
    plt.show()

# 保存结果为SEGY文件
def save_as_segy(data, save_path, dx=1000, dt=1000, sc=10000):
    print(f"Saving denoised data to {save_path}...")
    create_segy(data, save_path, dx=dx, dt=dt, sc=sc)

# 主函数
if __name__ == "__main__":
    # 参数设置
    data_path = '../data/sgy_data/F.sgy'
    model_path = r"E:\DCTformer\model\save\latest_model_epoch99.pth"
    save_path = r"E:\DCTformer\data\sgy_data\F_fqk.sgy"
    save_path_noisepre=r"E:\DCTformer\data\sgy_data\F_noise_fqk.sgy"
    patch_size = 256
    stride_x = patch_size // 2
    stride_y = patch_size // 2
    batch_size = 22

    # 检测是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    seismic_noise, seismic_block_h, seismic_block_w = load_seismic_data(data_path)

    # 数据归一化
    seismic_noise, seismic_noise_max = normalize_data(seismic_noise)

    # 数据填充
    seismic_noise_padded, pad_h, pad_w = pad_data(seismic_noise, patch_size)

    # 数据切分
    patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut_patches(seismic_noise_padded, patch_size, stride_x, stride_y)

    # 加载模型
    model = load_model(model_path, device)

    # 去噪处理
    predict_datas = denoise_patches(patches, model, device, batch_size)

    # 数据还原
    seismic_predict_padded = combine_patches(predict_datas, patch_size, strides_x, strides_y, seismic_block_h + pad_h, seismic_block_w + pad_w, stride_x, stride_y)

    # 数据裁剪
    seismic_predict = crop_data(seismic_predict_padded, seismic_block_h, seismic_block_w)

    # 数据逆归一化
    seismic_predict = unnormalize_data(seismic_predict, seismic_noise_max)
    seismic_noise_renorm=unnormalize_data(seismic_noise, seismic_noise_max)
    # 可视化结果
    noise_predict=seismic_noise_renorm-seismic_predict
    visualize_results(seismic_noise_renorm, seismic_predict, noise_predict)

    # 保存结果为SEGY文件
    save_as_segy(seismic_predict, save_path)
    save_as_segy(noise_predict, save_path_noisepre)