# -*-coding:utf-8-*-

# '''
#     1.对原始数据块(arr1)的右方和下方进行填充，使其横向和竖向都可以整除patch(L*L)。
#     2.将切好的patch喂入网络训练后，只取数据的中心部分(L * L),按照顺序拼起来既可以和arr1一样大的数据。
# '''
import numpy as np
import numpy as np
from scipy.signal import windows
def cut(seismic_block, patch_size, stride_x, stride_y):
    """
    :param seismic_block: 地震数据
    :param patch_size: 切片大小
    :param stride_x: 横向切片步长，小于patch_size以增加重叠
    :param stride_y: 竖向切片步长，小于patch_size以增加重叠
    :return: 按照规则填充后，获得的切片数据(以列表形式存储)，高方向切片数量，宽方向切片数量
    """
    [seismic_h, seismic_w] = seismic_block.shape
    patches = []
    for i in range(0, seismic_h - patch_size + 1, stride_y):
        for j in range(0, seismic_w - patch_size + 1, stride_x):
            patch = seismic_block[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return patches, (seismic_h - patch_size) // stride_y + 1, (seismic_w - patch_size) // stride_x + 1, seismic_h, seismic_w


def combine(patches, patch_size, number_h, number_w, block_h, block_w, stride_x, stride_y):
    combined = np.zeros((block_h, block_w), dtype=np.float32)
    count_matrix = np.zeros((block_h, block_w), dtype=np.int32)

    for i in range(number_h):
        for j in range(number_w):
            patch = patches[i * number_w + j]
            start_h = i * stride_y
            start_w = j * stride_x
            combined[start_h:start_h + patch_size, start_w:start_w + patch_size] += patch
            count_matrix[start_h:start_h + patch_size, start_w:start_w + patch_size] += 1

    # 检查并处理 count_matrix 中的零值
    count_matrix[count_matrix == 0] = 1  # 将零值替换为1

    combined = combined / count_matrix

    return combined

def combine_gaussian(patches,
                     patch_size,
                     number_h,
                     number_w,
                     block_h,
                     block_w,
                     stride_x,
                     stride_y,
                     sigma_ratio=6):
    """
    使用高斯加权平均的方式还原整幅图像，消除 patch 边界痕迹。
    参数说明与 combine() 完全一致，额外多一个可选参数 sigma_ratio:
        sigma_ratio: 高斯窗标准差 = patch_size / sigma_ratio，默认 4。
    """
    # ------------------ 构造二维高斯权重窗 ------------------
    sigma = patch_size / sigma_ratio
    gauss1d = windows.gaussian(patch_size, std=sigma)
    gauss2d = np.outer(gauss1d, gauss1d).astype(np.float32)

    # ------------------ 初始化叠加缓冲 ------------------
    combined = np.zeros((block_h, block_w), dtype=np.float32)
    count_matrix = np.zeros_like(combined)   # 权重累加器

    # ------------------ 逐 patch 叠加 ------------------
    for i in range(number_h):
        for j in range(number_w):
            patch = patches[i * number_w + j]
            start_h = i * stride_y
            start_w = j * stride_x
            end_h, end_w = start_h + patch_size, start_w + patch_size

            # 边界保护，防止越界
            if end_h > block_h or end_w > block_w:
                # 裁剪 patch 与窗口到实际剩余区域
                ph = min(patch_size, block_h - start_h)
                pw = min(patch_size, block_w - start_w)
                patch = patch[:ph, :pw]
                g_win = gauss2d[:ph, :pw]
            else:
                g_win = gauss2d

            combined[start_h:end_h, start_w:end_w] += patch * g_win
            count_matrix[start_h:end_h, start_w:end_w] += g_win

    # ------------------ 归一化 ------------------
    # 理论上 count_matrix 不会为 0，但保险起见
    count_matrix[count_matrix == 0] = 1.0
    combined /= count_matrix

    return combined