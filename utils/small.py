import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ====== 数据路径 ======
img_dir = '../datasets/TrainDataset/Imgs'   # 原图路径
# mask_dir = '../datasets/CodDataset/train/Scribble_wGT'    # 掩码路径
# mask_dir = '../datasets/TrainDataset/GT'    # 掩码路径
mask_dir = '../datasets/TrainDataset/oldScribble'    # 掩码路径
# ====== 输出路径 ======
for i, dataset in enumerate([
    'SmallCOD1', 'SmallCOD3', 'SmallCOD5', 'SmallCOD7', 'SmallCOD10','SmallCOD20',
    'SmallCOD30', 'SmallCOD40', 'SmallCOD50', 'SmallCOD60', 'SmallCOD70', 'SmallCOD80', 'SmallCOD90', 'SmallCOD100']):
    save_root = f'../datasets/TrainDataset/SmallCOD/{dataset}/'
    # small_mask_dir = os.path.join(save_root, 'GT')
    # small_img_dir = os.path.join(save_root, 'Imgs')

# 创建文件夹
    for d in [save_root]:
        os.makedirs(d, exist_ok=True)

# ====== 参数设置 ======
    if dataset == 'SmallCOD1':
        area_threshold_ratio_l = 0.00
        area_threshold_ratio_r = 0.01
    elif dataset == 'SmallCOD3':
        area_threshold_ratio_l = 0.01
        area_threshold_ratio_r = 0.03
    elif dataset == 'SmallCOD5':
        area_threshold_ratio_l = 0.03
        area_threshold_ratio_r = 0.05
    elif dataset == 'SmallCOD7':
        area_threshold_ratio_l = 0.05
        area_threshold_ratio_r = 0.07
    elif dataset == 'SmallCOD10':
        area_threshold_ratio_l = 0.07
        area_threshold_ratio_r = 0.1
    elif dataset == 'SmallCOD20':
        area_threshold_ratio_l = 0.1  # 区间下界
        area_threshold_ratio_r = 0.2  # 区间上界
    elif dataset == 'SmallCOD30':
        area_threshold_ratio_l = 0.2
        area_threshold_ratio_r = 0.3
    elif dataset == 'SmallCOD40':
        area_threshold_ratio_l = 0.3
        area_threshold_ratio_r = 0.4
    elif dataset == 'SmallCOD50':
        area_threshold_ratio_l = 0.4
        area_threshold_ratio_r = 0.5
    elif dataset == 'SmallCOD60':
        area_threshold_ratio_l = 0.5
        area_threshold_ratio_r = 0.6
    elif dataset == 'SmallCOD70':
        area_threshold_ratio_l = 0.6
        area_threshold_ratio_r = 0.7
    elif dataset == 'SmallCOD80':
        area_threshold_ratio_l = 0.7
        area_threshold_ratio_r = 0.8
    elif dataset == 'SmallCOD90':
        area_threshold_ratio_l = 0.8
        area_threshold_ratio_r = 0.9
    else:
        area_threshold_ratio_l = 0.9
        area_threshold_ratio_r = 1.0

# ====== 结果列表 ======
    small_list = []

# ====== 开始筛选 ======
    for mask_name in tqdm(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, mask_name)
        img_path = os.path.join(img_dir, os.path.splitext(mask_name)[0] + '.jpg')

        if not os.path.exists(img_path):
            continue
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]
        if mask is None:
            continue
        mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
        # 获取前景点坐标
        fg_row_indices, fg_col_indices = np.where(mask[:, :, 0] == 1)
        points = list(zip(fg_col_indices, fg_row_indices))

        if len(points) == 0:
            points = np.array([])
            continue
        points = np.array(points)

        # 获取坐标范围 + 计算九宫格尺寸 【等价原逻辑，写法一致】
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        grid_width = (x_max - x_min)
        grid_height = (y_max - y_min)

        # 二值化
        # _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        #
        # total_area = binary.size
        # target_area = np.count_nonzero(binary)
        # --- 在区间 0.01 ~ 0.03 内的保存 ---
        # ratio = target_area / total_area
        ratio = grid_width * grid_height / 589824
        if area_threshold_ratio_l <= ratio < area_threshold_ratio_r:
            # shutil.copy(mask_path, small_mask_dir)
            # shutil.copy(img_path, small_img_dir)
            small_list.append(mask_name)
# ====== 结果统计 ======
    print(f"✅ 小目标图像数量: {len(small_list)}")

# ====== 保存文件名列表 ======
    with open(os.path.join(save_root, 'SmallCOD.txt'), 'w') as f:
        f.writelines([n + '\n' for n in small_list])

    print("\n🎯 筛选完成！结果已保存到：", save_root)
