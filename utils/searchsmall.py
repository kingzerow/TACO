import os
import shutil
from tqdm import tqdm

# 文件路径 这段代码的主要功能是从一个大的数据集中筛选出特定的小目标图像，
# for _data_name in ['SmallCOD1', 'SmallCOD3', 'SmallCOD5', 'SmallCOD7', 'SmallCOD10', 'SmallCOD20']:
# for _data_name in ['SmallCOD1', 'SmallCOD3', 'SmallCOD5', 'SmallCOD7', 'SmallCOD10', 'SmallCOD20', 'SmallCOD30']:
# for _data_name in [ 'SmallCOD30','SmallCOD40', 'SmallCOD50', 'SmallCOD60', 'SmallCOD70', 'SmallCOD80']:
for _data_name in ['SmallCOD30','SmallCOD40', 'SmallCOD50', 'SmallCOD60', 'SmallCOD70', 'SmallCOD80']:
#     for _data_path in ['P=2', 'K=10_P=5', 'K=10_P=10', 'K=10_P=20', 'K=10_P=30']:
    for _data_path in ['P=15']:
    # small_list_file = "../datasets/TrainDataset/{}.txt".format(_data_name)  # 小目标文件名列表
        small_list_file = "../datasets/TrainDataset/SmallCOD/{}/SmallCOD.txt".format(_data_name)  # 小目标文件名列表
# for _data_name in ['CamoFormer-P', 'FSPNet', 'HDPNet', 'RISNet', 'RUN', 'VSCode', 'ZoomNet']:
    # src_dir = "/mnt/d/数据集/cod_results/{}/COD10K/".format(_data_name)        # 要筛选的源目录{}/GT/'.format(_data_name)
    # save_dir = "../PySODEvalToolkit-master/Data/SmallCOD10/Mask/{}/".format(_data_name) # 保存小目标对应图片的目录
    # src_dir = '../results/BGPL_COD2/EFEM_RFD_EHM/bs24_t1_pvtb4_EFEM_RFD_EHM_watt/COD10K_S/'        # 要筛选的源目录
    # save_dir = '../results/BGPL_COD_small/EFEM_RFD_EHM/bs24_t1_pvtb4_EFEM_RFD_EHM_watt/' # 保存小目标对应图片的目录
    #     src_dir = '../datasets/TrainDataset/{}/GT/'.format(_data_path)        # 要筛选的源目录
        src_dir = '/home/16t/wrs/datasets/TrainDataset/ScribbleRobustness/P=5/retain_25pct'.format(_data_path)        # 要筛选的源目录
        # src_dir = '/home/16t/wrs/datasets/TrainDataset/ScribbleRobustness/retain_25pct'.format(_data_path)        # 要筛选的源目录
        # save_dir = '../datasets/TrainDataset/{}/{}'.format(_data_path, _data_name) # 保存小目标对应图片的目录
        # save_dir = '../datasets/TrainDataset/mix{10,15}_ref_label_smallsize_SmallCOD1_r=0.2/GT' # 保存小目标对应图片的目录
        # save_dir = '/home/16t/wrs/datasets/TrainDataset/mix{10,15}_r=0.2/GT'
        save_dir = '/home/16t/wrs/datasets/TrainDataset/ScribbleRobustness/mix{5,10}_r=0.2_retain_25pct/GT'
        os.makedirs(save_dir, exist_ok=True)

        # 读取小目标文件名（去掉换行符）
        with open(small_list_file, "r") as f:
            small_files = [x.strip() for x in f.readlines() if x.strip()]
        print(len(small_files))
        # 批量匹配并复制
        for name in tqdm(small_files):
            # 统一后缀（若掩码是 .png 而原图是 .jpg，可自动替换）
            base_name = os.path.splitext(name)[0]
            for ext in [".png"]:
                src_path = os.path.join(src_dir, base_name + ext)
                if os.path.exists(src_path):
                    shutil.copy(src_path, os.path.join(save_dir, base_name + ext))
                    # shutil.copy(src_path, os.path.join(save_dir, base_name))
                    break
                else:
                    print(os.getcwd())
                    print("File not found:", src_path)