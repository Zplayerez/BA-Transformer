import cv2
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import json


def process_isic2018(
        dim=(352, 352), 
        input_dir='/home/ubuntu/Workspace/isic2018',
        save_dir='./ISIC2018/'):
    """处理isic2018数据集，支持png格式的图像和掩码"""
    
    # 训练集目录
    train_image_dir = os.path.join(input_dir, 'train/images')
    train_mask_dir = os.path.join(input_dir, 'train/masks')
    
    # 测试集目录
    test_image_dir = os.path.join(input_dir, 'test/images')
    test_mask_dir = os.path.join(input_dir, 'test/masks')
    
    # 确保输出目录存在
    os.makedirs(os.path.join(save_dir, 'Image'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'Label'), exist_ok=True)
    
    # 处理训练集图像
    process_folder(train_image_dir, train_mask_dir, save_dir, dim)
    
    # 处理测试集图像
    process_folder(test_image_dir, test_mask_dir, save_dir, dim)
    
    # 创建数据集分割文件
    create_dataset_split(save_dir)


def process_folder(image_dir, mask_dir, save_dir, dim):
    """处理单个文件夹中的图像和掩码"""
    
    # 获取所有图像文件
    image_path_list = os.listdir(image_dir)
    image_path_list = list(filter(lambda x: x.lower().endswith('.png'), image_path_list))
    image_path_list.sort()
    
    print(f"处理 {image_dir} 中的 {len(image_path_list)} 个图像...")
    
    for image_path in tqdm(image_path_list):
        # 图像ID
        _id = os.path.splitext(image_path)[0]
        
        # 构建掩码路径（假设掩码和图像有相同的文件名）
        mask_path = os.path.join(mask_dir, image_path)
        image_path = os.path.join(image_dir, image_path)
        
        # 确保掩码文件存在
        if not os.path.exists(mask_path):
            print(f"警告：掩码文件 {mask_path} 不存在，跳过该图像")
            continue
        
        # 读取图像和掩码
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        
        if image is None or mask is None:
            print(f"警告：无法读取图像或掩码 {image_path}，跳过")
            continue
        
        # 调整大小
        image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
        image_new = np.array(image_new, dtype=np.uint8)
        
        mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)
        # 二值化掩码
        mask_new = (mask_new > 127).astype(np.uint8) * 255
        
        # 保存为npy格式
        save_image_path = os.path.join(save_dir, 'Image', f"{_id}.npy")
        save_mask_path = os.path.join(save_dir, 'Label', f"{_id}.npy")
        
        np.save(save_image_path, image_new)
        np.save(save_mask_path, mask_new / 255.0)  # 归一化掩码到 0-1 范围


def create_circular_mask(h, w, center, radius):
    """创建圆形掩码"""
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask


def draw_msra_gaussian(heatmap, center, sigma):
    """在热图上绘制高斯分布"""
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def generate_point_annotations(save_dir):
    """为数据集生成点标注"""
    label_dir = os.path.join(save_dir, 'Label')
    point_dir = os.path.join(save_dir, 'Point')
    os.makedirs(point_dir, exist_ok=True)
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    
    print(f"为 {len(label_files)} 个掩码生成点标注...")
    
    for label_file in tqdm(label_files):
        _id = os.path.splitext(label_file)[0]
        label_path = os.path.join(label_dir, label_file)
        
        # 加载掩码
        label = np.load(label_path)
        label_ori = label.copy()
        
        # 将掩码转换为二值图像
        label_binary = (label > 0.5).astype(np.uint8) * 255
        
        # 查找轮廓
        contours, _ = cv2.findContours(label_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 创建点热图
        point_heatmap = np.zeros((352, 352))
        
        if len(contours) > 0:
            for contour in contours:
                stds = []
                points = contour[:, 0]  # (N,2)
                points_number = contour.shape[0]
                
                if points_number < 30:
                    continue
                
                # 根据轮廓点数设置参数
                if points_number < 100:
                    radius = 6
                    neighbor_points_n_oneside = 3
                elif points_number < 200:
                    radius = 10
                    neighbor_points_n_oneside = 15
                elif points_number < 300:
                    radius = 10
                    neighbor_points_n_oneside = 20
                elif points_number < 350:
                    radius = 15
                    neighbor_points_n_oneside = 30
                else:
                    radius = 10
                    neighbor_points_n_oneside = 40
                
                # 计算每个点的特征
                for i in range(points_number):
                    mask = create_circular_mask(352, 352, points[i], radius)
                    overlap_area = np.sum(mask * label_ori) / (np.pi * radius * radius)
                    stds.append(overlap_area)
                
                # 选择关键点
                selected_points = []
                stds = np.array(stds)
                
                for i in range(len(points)):
                    neighbor_points_index = np.concatenate([
                        np.arange(-neighbor_points_n_oneside, 0),
                        np.arange(1, neighbor_points_n_oneside + 1)
                    ]) + i
                    
                    # 处理环形数组下标
                    neighbor_points_index[np.where(neighbor_points_index < 0)[0]] += len(points)
                    neighbor_points_index[np.where(neighbor_points_index > len(points) - 1)[0]] -= len(points)
                    
                    # 选择具有局部最大或最小特征的点
                    if stds[i] < np.min(stds[neighbor_points_index]) or stds[i] > np.max(stds[neighbor_points_index]):
                        point_heatmap = draw_msra_gaussian(point_heatmap, (points[i, 0], points[i, 1]), 5)
                        selected_points.append(points[i])
        
        # 保存点热图
        save_point_path = os.path.join(point_dir, f"{_id}.npy")
        np.save(save_point_path, point_heatmap)


def create_dataset_split(save_dir, k=5):
    """创建数据集的k折交叉验证分割"""
    image_dir = os.path.join(save_dir, 'Image')
    indexes = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.npy')]
    
    # 创建K折交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    val_index = {}
    for i in range(k):
        val_index[str(i)] = []
    
    for i, (train_idx, val_idx) in enumerate(kf.split(indexes)):
        for idx in val_idx:
            val_index[str(i)].append(indexes[idx])
        print(f'fold:{i}, train_len:{len(train_idx)}, val_len:{len(val_idx)}')
    
    # 确保dataset目录存在
    os.makedirs('dataset', exist_ok=True)
    
    # 保存分割信息
    with open('dataset/data_split.json', 'w') as f:
        json.dump(val_index, f)
    
    print("数据集分割文件已保存到 dataset/data_split_isic2018.json")


if __name__ == '__main__':
    # 处理数据集
    process_isic2018(
        dim=(352, 352),
        input_dir='/home/ubuntu/Workspace/isic2018',
        save_dir='./ISIC2018/')
    
    # 生成点标注
    generate_point_annotations('./ISIC2018/')
