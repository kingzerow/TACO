import os

# 关键：修改为你的图片文件夹路径（绝对路径/相对路径都可以）
folder_path = "../datasets/TrainDataset/Depth"

# 遍历文件夹内所有文件
for filename in os.listdir(folder_path):
    # 只处理开头是processed_、后缀是.jpg的文件（适配你的需求）
    if filename.startswith("processed_") and filename.endswith(".jpg"):
        # 去掉开头的processed_
        new_filename = filename.replace("processed_", "", 1)  # 第三个参数1：只替换开头1次
        # 拼接原文件路径和新文件路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        # 重命名
        os.rename(old_path, new_path)
        print(f"已修改：{filename} → {new_filename}")

print("批量重命名完成！")