import os
import numpy as np
from scipy.io import loadmat, savemat

def cut_and_save_mat(input_folder, output_folder, block_size=256):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化文件名计数器
    file_counter = 1

    # 遍历输入文件夹中的所有.mat文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.mat'):
            # 加载.mat文件
            file_path = os.path.join(input_folder, filename)
            mat_data = loadmat(file_path)
            print(f"Processing {filename}...")

            # 假设.mat文件中有一个主要的矩阵变量（需要根据实际情况调整）
            for var_name, var_value in mat_data.items():
                if isinstance(var_value, np.ndarray) and var_value.ndim == 2:
                    print(f"Processing variable {var_name} with shape {var_value.shape}...")

                    # 获取矩阵的维度
                    rows, cols = var_value.shape

                    # 计算可以切割的块的数量
                    num_blocks_row = (rows + block_size - 1) // block_size
                    num_blocks_col = (cols + block_size - 1) // block_size

                    # 遍历每个块
                    for i in range(num_blocks_row):
                        for j in range(num_blocks_col):
                            # 计算块的边界
                            start_row = i * block_size
                            end_row = min(start_row + block_size, rows)
                            start_col = j * block_size
                            end_col = min(start_col + block_size, cols)

                            # 提取块
                            block = var_value[start_row:end_row, start_col:end_col]

                            # 如果块小于256×256，填充到256×256（可选）
                            if block.shape[0] < block_size or block.shape[1] < block_size:
                                padded_block = np.zeros((block_size, block_size), dtype=block.dtype)
                                padded_block[:block.shape[0], :block.shape[1]] = block
                                block = padded_block

                            # 保存块到新的.mat文件
                            block_filename = f"{file_counter}.mat"  # 使用计数器生成文件名
                            block_path = os.path.join(output_folder, block_filename)
                            savemat(block_path, {'Y2': block})  # 将数据保存到键 'Y2' 中
                            print(f"Saved block to {block_path}")

                            # 更新文件名计数器
                            file_counter += 1

    print("Processing complete.")

# 示例用法
input_folder = "D:/noisy/target"  # 替换为你的输入文件夹路径
output_folder = "D:/noisy/256/target"  # 替换为你的输出文件夹路径
cut_and_save_mat(input_folder, output_folder)