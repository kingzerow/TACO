from torch.utils.data import DataLoader
import os
import builtins
import argparse
import torch
import torch.distributed as dist
import time
import utils.metrics as Measure
import torch.nn.functional as F
import numpy as np
# from datasets import dataset
from utils.dataloader_text import test_dataset, encode_text_batch, load_text_encoder
from model.TACO import MyNet as CamoFormer
from utils.tools import *
from PIL import Image


path = 'xxx'
data_path = 'xxx'
path = 'xxx'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--path', type=str, default='../datasets', help='path to train dataset')
    parser.add_argument('--pretrain', type=str,
                        default='./checkpoints/{}/TACO/{}/Net_epoch_best.pth'.format(data_path, path),
                        help='path to pretrain model')
    parser.add_argument('--save_path', type=str, default='../results/{}/TACO/{}_blip'.format(data_path, path))
    # DDP configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank for distributed training')
    args = parser.parse_args()
    return args


def get_transform(ops=[0, 1, 2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op == 0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op == 1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op == 2:
        pp = Crop(0.7, 0.7)
    return pp


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(3407)


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            print("args.rank = {}; args.gpu = {}".format(args.rank, args.gpu))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not on master gpu
    if args.rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    ### model ###
    net = CamoFormer()
    
    if args.pretrain:
        device = torch.device('cuda:0')
        key = torch.load(args.pretrain, map_location=device)
        state_dict = key if 'state_dict' not in key else key['state_dict']
        net.load_state_dict(state_dict, strict=False)
        # net.load_state_dict(key['state_dict'], strict=True)
    if args.distributed:

        os.system("nvidia-smi")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net = net.cuda(args.gpu)
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        net.cuda()
    # ================== 数据集加载 ==================
    TestDir = ['../datasets/TestDataset/COD10K']
    # TestDir = ['../datasets/TestDataset/CAMO', '../datasets/TestDataset/COD10K', '../datasets/TestDataset/NC4K']
    text_encoder_type = 'clip-large'
    text_encoder = load_text_encoder(text_encoder_type)

    TestDataset = [
        test_dataset(
            image_root=test_path + '/Imgs/',
            gt_root=test_path + '/GT/',
            # text_json_path=test_path + '/test_with_label.json',
            # text_json_path=test_path + '/test_with_label_qwen.json',
            text_json_path=test_path + '/test_with_label_blip.json',
            # key="class_and_summary",# summary, class_and_summary, other
            # key="summary_qwen",# summary, class_and_summary, other
            key="summary",# summary, class_and_summary, other,summary_blip
            testsize=384,
            text_encoder_type=text_encoder_type
        ) 
        for test_path in TestDir
    ]
    TestDataloaders = [DataLoader(v, batch_size=1) for v in TestDataset]

    for i, TestDataloader in enumerate(TestDataloaders):
        net.train(False)
        FM = Measure.Fmeasure()
        WFM = Measure.WeightedFmeasure()
        SM = Measure.Smeasure()
        EM = Measure.Emeasure()
        MAE = Measure.MAE()
        results_all = []
        with torch.no_grad():
            st = time.time()
            for data in TestDataloader:
                image, mask, name = data['image'].cuda(), data['gt'], data['name']
                test_text_feat = encode_text_batch(
                    tokenized_batch={
                        "input_ids": data["input_ids"],
                        "attention_mask": data["attention_mask"]
                    },
                    text_encoder=text_encoder
                )
                print('=>{} eval start!'.format(name[0]))
                P1, P2, P3 = net(image, test_text_feat)
                out = F.interpolate(P2, size=mask.size()[2:], mode='bilinear', align_corners=False)
                pred = out.sigmoid().data.cpu().numpy().squeeze()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)
                mask = mask.numpy().astype(np.float32).squeeze()
                mask /= (mask.max() + 1e-8)
                pred_image = Image.fromarray((pred * 255).astype(np.uint8))
                save_path = args.save_path + '/' + TestDataloader.dataset.data_name.split('/')[-1]
                # if i == 0:
                #     save_path = save_path + 'CAMO/'
                # if i == 1:
                #     save_path = save_path + 'COD10K/'
                # if i == 2:
                #     save_path = save_path + 'NC4K/'
                # save_path = save_path + 'COD10K/'
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                # pred_image.save(save_path +'/' + name[0].split('/')[-1])
                FM.step(pred=pred * 255, gt=mask * 255)
                WFM.step(pred=pred* 255, gt=mask* 255)
                SM.step(pred=pred* 255, gt=mask* 255)
                EM.step(pred=pred* 255, gt=mask* 255)
                MAE.step(pred=pred* 255, gt=mask* 255)
                fm = FM.get_results()["fm"]
                wfm = WFM.get_results()["wfm"]
                sm = SM.get_results()["sm"]
                em = EM.get_results()["em"]
                mae = MAE.get_results()["mae"]
            results = {
                "Smeasure": sm,
                "wFmeasure": wfm,
                "meanEm": em["curve"].mean(),
                "adpFm": fm["adp"],
                "MAE": mae,
                "adpEm": em["adp"],
                "maxEm": em["curve"].max(),
                "meanFm": fm["curve"].mean(),
                "maxFm": fm["curve"].max(),
            }
            print(results)
            results_all.append(results)
            with open('../results/{}/results.txt'.format(data_path),"a")as f :
                f.write(str(results) +'{}{}:'.format(TestDataloader.dataset.data_name, path)+'P2_class_and_summary\n')
            print('dataset: {}, spend_time:{}'.format(
                TestDataloader.dataset.data_name, (time.time() - st)))
        net.train(True)
    # print('maes:{}'.format(maes))
    print('results_all:{}'.format(results_all))
if __name__ == '__main__':
    args = parse_args()
    main(args)