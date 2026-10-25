import os
import shutil

from tqdm import tqdm


# 第一组图像的源目录
src_dir_a = (
    "/home/16t/wrs/datasets/TrainDataset/"
    "ScribbleRobustness/P=5/retain_75pct"
)

# 第二组图像的源目录
src_dir_b = (
    "/home/16t/wrs/datasets/TrainDataset/"
    "ScribbleRobustness/retain_75pct"
)

# 两组图像统一保存到该目录
save_dir = (
    "/home/16t/wrs/datasets/TrainDataset/"
    "ScribbleRobustness/mix{5,10}_r=0.2_retain_75pct/GT"
)

group_a = [
    "SmallCOD1",
    "SmallCOD3",
    "SmallCOD5",
    "SmallCOD7",
    "SmallCOD10",
    "SmallCOD20",
]

group_b = [
    "SmallCOD30",
    "SmallCOD40",
    "SmallCOD50",
    "SmallCOD60",
    "SmallCOD70",
    "SmallCOD80",
]


def copy_selected_images(src_dir, data_names, copied_files, missing_files):
    """根据 SmallCOD.txt，从 src_dir 筛选图像并复制到 save_dir。"""
    for data_name in data_names:
        list_file = (
            f"/home/user1/wrs/datasets/TrainDataset/SmallCOD/{data_name}/SmallCOD.txt"
        )

        with open(list_file, "r", encoding="utf-8") as file:
            image_names = [line.strip() for line in file if line.strip()]

        print(f"{data_name}: {len(image_names)} images")

        for image_name in tqdm(image_names, desc=data_name):
            base_name = os.path.splitext(image_name)[0]
            file_name = base_name + ".png"

            # 两组清单存在相同图像时只复制一次。
            if file_name in copied_files:
                continue

            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(save_dir, file_name)

            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                copied_files.add(file_name)
            else:
                missing_files.append(src_path)


def main():
    os.makedirs(save_dir, exist_ok=True)

    copied_files = set()
    missing_files = []

    # 从 src_dir_a 筛选第一组。
    copy_selected_images(
        src_dir_a,
        group_a,
        copied_files,
        missing_files,
    )

    # 从 src_dir_b 筛选第二组。
    copy_selected_images(
        src_dir_b,
        group_b,
        copied_files,
        missing_files,
    )

    print(f"Copied {len(copied_files)} images to: {save_dir}")

    if missing_files:
        print(f"File not found ({len(missing_files)}):")
        for path in missing_files:
            print(path)


if __name__ == "__main__":
    main()
