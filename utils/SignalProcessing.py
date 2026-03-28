# -*-coding:utf-8-*-
"""
Created on 2022.3.5
programing language:python
@author:夜剑听雨
"""
import numpy as np
import math
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn.functional as F
# utils/SignalProcessing.py
import torch
import torch_dct as dct  # pip install torch-dct

def compare_SNR(recov_img, real_img):
    """
    计算信噪比
    :param recov_img:重建后或者含有噪声的数据
    :param real_img: 干净的数据
    :return: 信噪比
    """
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var == 0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var, 10)
    return s
def batch_snr(de_data, clean_data):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = de_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :, :].squeeze()
        SNR += compare_SNR(De, Clean)
    return SNR / De_data.shape[0]

def mse(signal, noise_data):
    """
    计算均方误差
    Args:
        signal: 信号
        noise_data: 含噪声数据
    Returns:均方误差
    """
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    m = np.sum((signal - noise_data) ** 2)  # numpy可以并行运算
    m = m / m.size  # mse.size输出矩阵的元素个数
    return m

def psnr(signal, noise_data):
    """
    计算峰值信噪比
    Args:
        signal: 信号
        noise_data: 含噪声数据
    Returns:峰值信噪比
    """
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    psnr = 2 * 10 * math.log10(abs(signal.max()) / np.sqrt(np.sum((signal - noise_data) ** 2) / noise_data.size))
    return psnr

def fft_spectrum(Signal, SampleRate):
    """
    计算一维信号的傅里叶谱
    :param Signal: 一维信号
    :param SampleRate: 采样率，一秒内的采样点数
    :return: 傅里叶变换结果
    """
    fft_len = Signal.size  # 傅里叶变换长度
    # 原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    SignalFFT = np.fft.rfft(Signal) / fft_len  # 变换后归一化处理
    SignalFreqs = np.linspace(0, SampleRate/2, int(fft_len/2)+1)  # 生成频率区间
    SignalAmplitude = np.abs(SignalFFT) * 2   # 复数的模代表的是振幅
    return SignalFreqs, SignalAmplitude

# 巴沃斯低通滤波器
def butter_lowpass(cutoff, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutoff = cutoff / rate
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(noise_data, cutoff, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 巴沃斯带通滤波器
def butter_bandpass(lowcut, highcut, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    low = lowcut / rate
    high = highcut / rate
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(noise_data, lowcut, highcut, sample_rate, order=4):
    """
    带通滤波器
    :param noise_data: 含噪声数据
    :param lowcut: 带通滤波的最小值
    :param higtcut: 带通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data
# 巴沃斯高通滤波器
def butter_highpass(cutup, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutup = cutup / rate
    b, a = signal.butter(order, normal_cutup, btype='high', analog=False)
    return b, a

def highpass_filter(noise_data, cutup, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_highpass(cutup, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 一维信号的中值滤波器
# python的中值滤波函数对数组的维数要求严格，打个比方你用维数为（200，1）的数组当输入，不行！
# 必须改成（200，才会给你滤波。
def mide_filter(x,kernel_size=5):
    """
    中值滤波器
    :param x: 一维信号
    :param kernel_size: 滤波器窗口，默认为5
    :return: 中值滤波后的数据
    """
    x1 = x.reshape(x.size)
    y = signal.medfilt(x1, kernel_size=kernel_size)
    return y

def fk_spectra(data, dt, dx, L=6):
    """
    f-k(频率-波数)频谱分析
    :param data: 二维的地震数据
    :param dt: 时间采样间隔
    :param dx: 道间距
    :param L: 平滑窗口
    :return: S(频谱结果), f(频率范围), k(波数范围)
    """
    data = np.array(data)
    [nt, nx] = data.shape  # 获取数据维度
    # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
    H1 = np.hamming(L)
    # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f


# SignalProcessing.py 中的 ssim_loss 函数,针对 PyTorch 张量

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                      range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(gt_list))]

    # 如果图像数据是归一化到 [0, 1] 的，不需要乘以 255
    # 如果图像数据是 [0, 255] 的，需要乘以 255
    # 这里假设图像数据是归一化到 [0, 1] 的
    ssim_list = [compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1) for ind in
                 range(len(dehaze_list))]

    return ssim_list


#二维数组张量ssim
def ssim_2d_np(img1, img2):
    """计算两幅 2D NumPy 数组（H×W）之间的 SSIM"""
    img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)  # 归一化到 [0, 1]
    img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
    return compare_ssim(img1, img2, data_range=1.0)

#DCT域的lloss
def dct_l1_loss(x, y, reduction='mean'):
    """
    先对输入 x, y 做 DCT 变换，再计算 L1 损失。
    输入 shape: (B, C, H, W)
    输出: 标量
    """
    # 把每个通道分别做 2D-DCT
    x_dct = dct.dct_2d(x, norm='ortho')
    y_dct = dct.dct_2d(y, norm='ortho')
    return torch.nn.functional.l1_loss(x_dct, y_dct, reduction=reduction)