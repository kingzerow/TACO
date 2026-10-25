import math
import random
from tqdm import tqdm
import numpy as np
import os
import glob
from PIL import Image
import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print("当前工作目录：", os.getcwd())
sam_checkpoint = "/home/user1/wrs/pretrained/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = 'cuda'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
np.random.seed(42)

import numpy as np

def nine_grid_sampling(points, pick_number=5):
    if len(points) == 0:
        return np.array([])
    points = np.array(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    grid_width = (x_max - x_min) / 3
    grid_height = (y_max - y_min) / 3

    grids = [[] for _ in range(9)]
    for p in points:
        gx = min(int((p[0] - x_min) / grid_width), 2)
        gy = min(int((p[1] - y_min) / grid_height), 2)
        idx = gy * 3 + gx  # 映射为0-8的一维索引
        grids[idx].append(p)

    total_number = len(points)
    number_proportions = np.zeros(9, dtype=int)
    if total_number > pick_number:
        for i in range(9):
            grid_point_num = len(grids[i])
            number_proportions[i] = int(np.ceil((grid_point_num / total_number) * pick_number))
    else:
        for i in range(9):
            number_proportions[i] = len(grids[i])

    sampled_points = []
    for i in range(9):
        grid_points = grids[i]
        np.random.shuffle(grid_points)
        pick_num = number_proportions[i]
        if pick_num > 0:
            sampled_points.append(np.array(grid_points[:pick_num]))


    sample_points = np.concatenate(sampled_points, axis=0) if sampled_points else np.array([])

    if len(sample_points) > pick_number:
        np.random.shuffle(sample_points)
        sample_points = sample_points[:pick_number]

    return sample_points


def get_mask(image, fig_mask, pick_number=5):
    np.random.seed(42)
    # get instance mask
    predictor.set_image(image)

    # 获取前景点坐标
    fg_row_indices, fg_col_indices = np.where(fig_mask[:, :, 0] == 1)
    fg_coordinates = list(zip(fg_col_indices, fg_row_indices))
    fg_sample_points = nine_grid_sampling(fg_coordinates, pick_number=pick_number)
    num_fg_points = len(fg_sample_points)
    fg_coordinates_label = np.ones(num_fg_points)

    # 获取背景点坐标
    bg_row_indices, bg_col_indices = np.where(fig_mask[:, :, 0] == 2)
    bg_coordinates = list(zip(bg_col_indices, bg_row_indices))
    bg_sample_points = nine_grid_sampling(bg_coordinates, pick_number=pick_number)
    # num_bg_points = len(bg_sample_points)
    # bg_coordinates_label = np.zeros(num_bg_points)
    if len(bg_sample_points) == 0:
        # 左上角当作背景点
        bg_sample_points = np.array([[1, 1]])
        bg_coordinates_label = np.array([0])
    else:
        bg_coordinates_label = np.zeros(len(bg_sample_points))

    coordinates = np.concatenate((fg_sample_points, bg_sample_points), axis=0)
    coordinates_labels = np.concatenate((fg_coordinates_label, bg_coordinates_label), axis=0)
    result = np.transpose(np.zeros_like(image), (2, 0, 1))
    for i in range(pick_number):
        coordinate = coordinates[i:i+1, :]
        coordinates_label = coordinates_labels[i:i+1]
        mask, scores, logits = predictor.predict(
            point_coords=coordinate,
            point_labels=coordinates_label,
            multimask_output=True,
        )

        coordinate = coordinates[:, :]
        coordinates_label = coordinates_labels[:]
        mask_input = logits[np.argmax(scores), :, :]
        mask, scores, logits = predictor.predict(
            point_coords=coordinate,
            point_labels=coordinates_label,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
        )
        result += mask

    result = result[0].astype(int)
    result = np.repeat(result[:, :, np.newaxis], 3, axis=2)
    return result


def get_random_mask(image_files, pick_number=5):
    random_number = random.randint(0, 7)

    sam_image = cv2.imread(image_files)
    mask_path = image_files.replace('Imgs', 'oldScribble').replace('jpg', 'png')
    mask = cv2.imread(mask_path).astype(np.float32)[:, :, ::-1]
    H1, W1, C1 = sam_image.shape
    sam_image = cv2.resize(sam_image, (768, 768), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
    H, W, C = sam_image.shape

    if random_number == 0:
        flipped_image = np.flip(sam_image, axis=1)
        flipped_mask = np.flip(mask, axis=1)
        flipped_mask = get_mask(flipped_image, flipped_mask, pick_number)
        return_mask = np.flip(flipped_mask, axis=1)

    if random_number >= 1 and random_number <= 3:
        # rotation
        if random_number == 1:
            angle = cv2.ROTATE_90_CLOCKWISE
            return_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif random_number == 2:
            angle = cv2.ROTATE_180
            return_angle = cv2.ROTATE_180
        elif random_number == 3:
            angle = cv2.ROTATE_90_COUNTERCLOCKWISE
            return_angle = cv2.ROTATE_90_CLOCKWISE
        rotated_image = cv2.rotate(sam_image, angle)
        rotated_mask = cv2.rotate(mask, angle)
        rotated_mask = get_mask(rotated_image, rotated_mask, pick_number)
        return_mask = cv2.rotate(rotated_mask, return_angle)

    if random_number == 4:
        return_mask = get_mask(sam_image, mask, pick_number)
    if random_number > 4:
        if random_number == 5:
            # 0.5
            scaling = 2
        if random_number == 6:
            # 1
            scaling = 1
        if random_number == 7:
            # 2
            scaling = 0.5
        # shrink
        normal_sam_image = cv2.resize(sam_image, (int(W / scaling), int(H / scaling)), interpolation=cv2.INTER_NEAREST)
        normal_mask = cv2.resize(mask, (int(W / scaling), int(H / scaling)), interpolation=cv2.INTER_NEAREST)
        normal_mask = get_mask(normal_sam_image, normal_mask, pick_number)
        return_mask = cv2.resize(normal_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return_mask = cv2.resize(return_mask, (W1, H1), interpolation=cv2.INTER_NEAREST)

    return return_mask, mask_path


if __name__ == '__main__':
    np.random.seed(42)
    # read image
    path = '../datasets/TrainDataset/Imgs'
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    K = 10
    pick_number = 5
    for image_file in tqdm(image_files):
        try:
            for k in range(K):
                if k == 0:
                    mask, mask_path = get_random_mask(image_file, pick_number)
                    final_mask = mask
                elif k == K - 1:
                    mask, mask_path = get_random_mask(image_file, pick_number)
                    final_mask = mask + final_mask
                    non_zero_values = final_mask[final_mask != 0]
                    average_non_zero = np.mean(non_zero_values)
                    final_mask[final_mask > average_non_zero/4] = 255
                    final_mask[final_mask <= average_non_zero/4] = 0
                else:
                    mask, _ = get_random_mask(image_file, pick_number)
                    final_mask = mask + final_mask

            final_mask = final_mask.astype(np.uint8)
            maskpath_save = mask_path.replace('oldScribble', 'P=5')
            mask_image = Image.fromarray(final_mask)
            mask_image.save(maskpath_save)
            # print(image_file)
        except:
            # output_txt_path = "../datasets/CodDataset/train/K=12.txt"

            # # 将单一的文件路径写入到 .txt 文件中
            # with open(output_txt_path, 'a') as f:
            #     f.write(str(image_files) + '\n')
            print(image_file, r'wrong')
            # break