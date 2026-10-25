import os


def find_files_a_not_in_b(folder_a: str, folder_b: str) -> list:
    """
    找出文件夹A中存在、但文件夹B中不存在的文件（仅对比直接文件，不递归子文件夹）
    :param folder_a: 源文件夹A的路径
    :param folder_b: 对比文件夹B的路径
    :return: A独有的文件的完整路径列表
    """
    # 校验文件夹是否存在
    if not os.path.isdir(folder_a):
        raise ValueError(f"文件夹A不存在：{folder_a}")
    if not os.path.isdir(folder_b):
        raise ValueError(f"文件夹B不存在：{folder_b}")

    # 获取文件夹B中的所有文件名（仅文件名，忽略路径）
    files_b = set()
    count_b = 0
    for file_name in os.listdir(folder_b):
        file_path = os.path.join(folder_b, file_name)
        if os.path.isfile(file_path):  # 仅处理文件，跳过子文件夹
            files_b.add(file_name)
            count_b += 1

    # 遍历文件夹A，筛选出B中没有的文件
    a_only_files = []
    count_a = 0
    for file_name in os.listdir(folder_a):
        file_path = os.path.join(folder_a, file_name)
        if os.path.isfile(file_path):  # 仅处理文件
            count_a += 1
            if file_name not in files_b:
                a_only_files.append(file_path)
    # 打印A、B的文件总数
    print(f"📊 文件夹文件数量统计：")
    print(f"   文件夹A（{folder_a}）总文件数：{count_a} 个")
    print(f"   文件夹B（{folder_b}）总文件数：{count_b} 个")

    return a_only_files


# #################### 用法示例 ####################
if __name__ == "__main__":
    # 替换为你的实际文件夹路径（Windows用r"C:\folder\a"，Linux/macOS用"/home/folder/a"）
    # FOLDER_A = '../datasets/TrainDataset/GT/'
    FOLDER_A = '../datasets/TrainDataset/mix1/GT/'
    FOLDER_B = '../datasets/TrainDataset/mix2/GT/'
    # FOLDER_B = '../datasets/TrainDataset/mix{10,15}_ref_label_smallsize_SmallCOD1_r=0.2/GT/'
    # FOLDER_B = '../datasets/TrainDataset/mix2_ref_label_smallsize/GT/'
    # FOLDER_B = '/home/user1/wrs/datasets/TrainDataset/mix2_ref_0.4/GT/'
    # FOLDER_B = '/home/16t/wrs/datasets/TrainDataset/mix{5,10}_r=0.2_fixed_0.3/GT'
    # FOLDER_B = '/home/user1/wrs/EVF-SAM-main/infer_batch/TextMask/'#/home/user1/wrs/datasets/TrainDataset/mix2_ref_0.4
    # FOLDER_B = '/home/user1/wrs/EVF-SAM-main/infer_batch/TextMask/'
    # FOLDER_A = '../datasets/TestDataset/COD10K/GT/'
    # FOLDER_B = '../datasets/TestDataset/COD10K/GT/'

    try:
        # 获取A独有的文件
        only_in_a = find_files_a_not_in_b(FOLDER_A, FOLDER_B)

        # 输出结果
        if only_in_a:
            print(f"文件夹A中存在、B中不存在的文件共 {len(only_in_a)} 个：")
            for file_path in only_in_a:
                print(f"  - {file_path}")
        else:
            print("文件夹A中的所有文件，在文件夹B中都能找到！")
    except Exception as e:
        print(f"执行出错：{e}")