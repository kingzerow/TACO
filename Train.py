#这个文件是测试文本内容
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/user1/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import random
import logging
from torch.utils.data import DataLoader
import builtins
import argparse
import torch.distributed as dist
import time
import torch
import utils.metrics as Measure
from datasets import dataset_edge as dataset
from utils.dataloader_text import test_dataset, get_loader, encode_text_batch, load_text_encoder
from utils import loss
from utils.utils import clip_gradient, AvgMeter
# from model.CamoFormer import CamoFormer
from model.TACO import MyNet as CamoFormer
from utils.tools import *
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
"""
GRENet7:TACO
GRENet11:w/o BGRM
GRENet13:w/o GCN
GRENet15:GCN->CNN
GRENet16:baseline

"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 1:blip 4:qwen
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 1:blip 4:qwen
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 1:blip 2:qwen
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # 1:blip 2:qwen
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.cloud.tencent.com"
def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def val(model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        test_loader = test_dataset(image_root=args.test_path + '/Imgs/',
                                   gt_root=args.test_path + '/GT/',
                                   testsize=args.trainsize)
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res, res1, _ = model(image)
            # res, res1 = model(image)
            # eval Dice
            res = F.upsample(res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print(test_loader.size)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        # if epoch == 1:
        #     best_mae = mae
        # else:
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
            print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        # CAMO

        # if mae < 0.044:
        #     torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))



train_text = 'other'
# train_text = 'summary_blip'

def parse_args():
    parser = argparse.ArgumentParser("FSPNet-Transformer")
    parser.add_argument('--base_lr', default=(1e-4), type=float, help='learning rate')
    parser.add_argument('--epoch', type=int, default=60, help='epoch number')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size per GPU')
    parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--path', type=str, default='../datasets', help='path to train dataset')
    parser.add_argument('--train_path', type=str, default='../datasets/TrainDataset', help='path to train dataset')
    # parser.add_argument('--train_path', type=str, default='/home/16t/wrs/datasets/TrainDataset', help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='../datasets/TestDataset/CAMO',
                        help='path to testing dataset')
    parser.add_argument('--pretrain', type=str, default='../pretrained/pvt_v2_b4.pth', help='path to pretrain model')
    parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')


    parser.add_argument('--save_path', type=str, default='/home/16t/wrs/mix{5,10}_r=0.2_fixed_0.3/GRENet7_ts384/')
    parser.add_argument('--dataset_path', type=str, default='mix{5,10}_r=0.2_fixed_0.3', help='save name')
    parser.add_argument('--train_text', type=str, default=train_text, help='save name')

    # DDP configs:Scribble_wGT
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for distributed training')

    parser.add_argument('--epoch_save', type=int, default=5, help='every n epochs to save model')

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


def recover_transform(out, value, shape):
    # 0 是 水平 1是垂直
    if value < 0.25:
        pp = Flip(0)
        out = pp(out)

    elif value < 0.5:
        pp = Flip(1)
        out = pp(out)

    elif value < 0.75:
        pp1 = Flip(0)
        pp2 = Flip(1)
        out = pp1(out)
        out = pp2(out)
    out = F.interpolate(out.unsqueeze(0), shape, mode='bilinear')
    return out
def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)   # 0.1* 200//100
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(args.save_path)
    logging.basicConfig(filename=args.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    set_seed(3407)
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
    if args.pretrain:
        model_name = os.path.splitext(os.path.basename(args.pretrain))[0]
        net = CamoFormer(model_name)
        print("当前工作路径:", os.getcwd())
        encoder = torch.load(args.pretrain)
        net.encoder.load_state_dict(encoder, strict=False)
    # net = FSPNet_model.Model(args.pretrain, img_size=384)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        os.system("nvidia-smi")
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net = net.cuda(args.gpu)
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    else:
        net.cuda()

    ### optimizer ###
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr)
    encoder_param = []
    decoer_param = []
    for name, param in net.named_parameters():
        if "encoder" in name:
            encoder_param.append(param)
        else:
            decoer_param.append(param)
    # optimizer = torch.optim.SGD([{"params": encoder_param, "lr":args.base_lr*0.1},{"params":decoer_param, "lr":args.base_lr}], momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(
        [{"params": encoder_param, "lr": args.base_lr * 0.1}, {"params": decoer_param, "lr": args.base_lr}])
    # optimizer = torch.optim.Adam(
        # [{"params": encoder_param, "lr": args.base_lr}, {"params": decoer_param, "lr": args.base_lr}])
    curr_epoch = 0
    ### resume training if necessary ###
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(ckpt['state_dict'])
        curr_epoch = ckpt['epoch']
        if curr_epoch >= args.decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1 * (curr_epoch / 30)
                print("Learning rate:", param_group['lr'])
        # optimizer.load_state_dict(ckpt['optimizer'])

    ### Fine tuning for MoCA ###
    # if args.ft_for_MoCA is not None:
    #     ckpt = torch.load(args.ft_for_MoCA, map_location='cpu')
    #     net.load_state_dict(ckpt)
    #     print("Fine tuning for MoCA, ckpt from: {}".format(args.ft_for_MoCA))
    print('model paramters', sum(p.numel() for p in net.parameters() if p.requires_grad))
    logging.info('model paramters: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    ### data ###
    # Dir = [args.path + '/TrainDataset']
    # Dataset = dataset.TrainDataset(Dir)
    # Datasampler = torch.utils.data.RandomSampler(Dataset, replacement=False)
    # train_loader = DataLoader(Dataset, batch_size=args.batch_size,
    #                         collate_fn=dataset.my_collate_fn, sampler=Datasampler, drop_last=True)
    image_root = '{}/Imgs/'.format(args.train_path)
    # gt_root = '{}/{}/GT/'.format(args.train_path, args.dataset_path)# mix2_ref_label_smallsize_SmallCOD1
    # edge_root = '{}/{}/Edge/'.format(args.train_path, args.dataset_path)# ../EVF-SAM-main/infer_ref_label_smallsize/TextMask/
    gt_root = "/home/16t/wrs/datasets/TrainDataset/{}/GT/".format(args.dataset_path)# mix2_ref_label_smallsize_SmallCOD1# gt_root = '../EVF-SAM-main/infer_ref_label_smallsize/TextMask/'
    edge_root = '/home/16t/wrs/datasets/TrainDataset/{}/Edge/'.format(args.dataset_path)# ../EVF-SAM-main/infer_ref_label_smallsize/TextMask/# edge_root = '../EVF-SAM-main/infer_ref_label_smallsize/Edge/'
    if args.train_text == 'summary':
        # train_root = 'datasets/train_with_label_with_size.json'
        train_root = 'datasets/train_with_caption_blip.json'
        cache_dir = './text_cache/ref4'
        TestDir = [args.path + '/TestDataset/CAMO', args.path + '/TestDataset/COD10K', args.path + '/TestDataset/NC4K']
    elif args.train_text == 'other':
        train_root = 'datasets/train_with_caption_blip.json'
        cache_dir = './text_cache/other'
        TestDir = [args.path + '/TestDataset/CAMO', args.path + '/TestDataset/COD10K', args.path + '/TestDataset/NC4K']
    elif args.train_text == 'summary_blip':
        train_root = 'datasets/train_with_caption_blip.json'
        cache_dir = './text_cache/blip'
        TestDir = [args.path + '/TestDataset/CAMO']
    elif args.train_text == 'summary_qwen':
        train_root = 'datasets/train_with_caption_qwen1.json'
        cache_dir = './text_cache/qwen'
        TestDir = [args.path + '/TestDataset/CAMO']
    # train_root = 'datasets/train_with_caption_blip.json'
    # train_root = 'datasets/train_with_caption_qwen1.json'
    # cache_dir = './text_cache/ref'
    # cache_dir = './text_cache/blip'
    text_encoder_type = "clip-large"  # 可选 "clip-base", clip-large, "bert-base", "roberta-base", "mamba-130m"
    train_loader = get_loader(image_root, gt_root, edge_root, train_root, key=args.train_text, batchsize=args.batch_size,
                              trainsize=args.trainsize, text_encoder_type=text_encoder_type, cache_dir=cache_dir)

    total_step = len(train_loader)
    print(total_step)
    # torch.backends.cudnn.benchmark = True
    # TestDir = [args.path + r'\TestDataset\CAMO', args.path + r'\TestDataset\COD10K',
    #            args.path + r'\TestDataset\CHAMELEON', args.path + r'\TestDataset\N4CK']

    # TestDir = [args.path + '/TestDataset/CAMO', args.path + '/TestDataset/COD10K', args.path + '/TestDataset/NC4K']
    # TestDir = [args.path + '/TestDataset/CAMO']

    # TestDataset = [test_dataset(v, size=416) for v in TestDir]
    # test_dataset(image_root=args.test_path + '/Imgs/',
    #                                gt_root=args.test_path + '/GT/',
    #                                testsize=args.trainsize)
    TestDataset = [
        test_dataset(
            image_root=test_path + '/Imgs/',
            gt_root=test_path + '/GT/',
            text_json_path=test_path + '/test_with_label.json',
            # text_json_path=test_path + '/ref_anno.json',
            # text_json_path=test_path + '/test_with_label_blip.json',
            # text_json_path=test_path + '/test_with_label_qwen.json',
            key=args.train_text,
            testsize=args.trainsize,
            text_encoder_type=text_encoder_type
        ) 
        for test_path in TestDir
    ]
    TestDataloaders = [DataLoader(v, batch_size=1) for v in TestDataset]

    writer = SummaryWriter(args.save_path + 'summary')
    best_mae = 3
    best_epoch = 0
    ### main loop ###
    loss_P1_record = AvgMeter()
    loss_P2_record = AvgMeter()
    loss_P3_record = AvgMeter()
    for epoch in tqdm(range(curr_epoch + 1, args.epoch + 1)):

        if epoch % args.decay_epoch == 0:
            torch.save({'state_dict': net.state_dict(),
                        'epoch': epoch
                        }, args.save_path + 'Net_epoch_{}.pth'.format(epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print("Learning rate:", param_group['lr'])
                writer.add_scalar('learning_rate', param_group['lr'], global_step=epoch)
        # cur_lr = adjust_lr(optimizer, args.base_lr, epoch, args.decay_rate, args.decay_epoch)
        # train
        net.train()
        running_loss_all, running_loss_m = 0., 0.
        # count = 0
        text_encoder = load_text_encoder(text_encoder_type)
        for i, data in enumerate(train_loader, start=1):
            # for data in train_loader:
            # count += 1
            # img, label, name, edge, strengthen, shape = data['img'].cuda(args.rank), data['label'].cuda(args.rank), data[
            #     'name'], data['edge'].cuda(args.rank), data['strengthen'], data['shape']
            # img, label, edge, text_features = data
            img = data["image"].cuda()
            label = data["gt"].cuda()
            edge = data["edge"].cuda()
            text_feat = encode_text_batch(
                tokenized_batch={
                    "input_ids": data["input_ids"],
                    "attention_mask": data["attention_mask"]
                },
                text_encoder=text_encoder
            )
            out1 = net(img, text_feat)
            all_loss1 = 1 * structure_loss(out1[0], label) + structure_loss(out1[1], label) + 1 * dice_loss(out1[2], edge)
            # all_loss1 = 3 * dice_loss(out1[0], edge) + structure_loss(out1[1], label) + structure_loss(out1[2], label)
            # all_loss1 = structure_loss(out1[0], label) + structure_loss(out1[1], label)
            pre_transform = get_transform([0, 1, 2])
            tr_image = pre_transform(img)
            tr_label = pre_transform(label)
            tr_edge = pre_transform(edge)
            tr_image = F.interpolate(tr_image, scale_factor=0.5, mode='bilinear', align_corners=True)
            tr_label = F.interpolate(tr_label, scale_factor=0.5, mode='bilinear', align_corners=True)
            tr_edge = F.interpolate(tr_edge, scale_factor=0.5, mode='bilinear', align_corners=True)
            out2 = net(tr_image, text_feat)
            # SaliencyStructure Consistancy
            out_s = pre_transform(out1[1])
            out_s = F.interpolate(out_s, scale_factor=0.5, mode='bilinear', align_corners=True)
            out_s_s = out2[1]
            loss_ssc = SaliencyStructureConsistency(out_s_s, out_s.detach(), 0.85)
            # loss_gradL1 = GradStructureConsistency(out_s_s, out_s.detach())
            # all_loss2, m_loss2 = loss.MSPNet_multi_bce(out2, tr_label)
            # all_loss = all_loss2 + all_loss1 + loss_ssc + loss_gradL1
            # m_loss = m_loss2 + m_loss1 + loss_ssc + loss_gradL1
            loss_p1 = structure_loss(out2[0], tr_label)
            loss_p2 = structure_loss(out2[1], tr_label)
            loss_edge = dice_loss(out2[2], tr_edge)
            all_loss2 = 1 * loss_p1 + loss_p2 + 1 * loss_edge
            all_loss = all_loss2 + all_loss1 + loss_ssc
            # m_loss = m_loss2 + m_loss1 + loss_ssc
            # m_loss = 0

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            loss_P1_record.update(loss_p1.data, args.batch_size)
            loss_P2_record.update(loss_p2.data, args.batch_size)
            # loss_P3_record.update(loss_edge.data, args.batch_size)
            running_loss_m += all_loss.item()
            writer.add_scalar('running_loss_all', running_loss_m, global_step=epoch)
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      ' lateral-1: [{:0.4f}], lateral-2: [{:0.4f}]'.
                      format(datetime.now(), epoch, args.epoch, i, total_step,
                             loss_P1_record.show(), loss_P2_record.show()))
                logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                             ' lateral-1: [{:0.4f}], lateral-2: [{:0.4f}]'.
                             format(datetime.now(), epoch, args.epoch, i, total_step,
                                    loss_P1_record.show(), loss_P2_record.show()))
        if epoch % args.epoch_save == 0:
            torch.save({'state_dict': net.state_dict(),
                        'epoch': epoch
                        }, args.save_path + 'Net_epoch_{}.pth'.format(epoch))
        torch.save({'state_dict': net.state_dict(),
                    'epoch': epoch
                    }, args.save_path + 'last.pth')

        if (epoch) % args.epoch_save == 0:
            maes = [1, 1, 1]
            for i, TestDataloader in enumerate(TestDataloaders):
                net.train(False)
                FM = Measure.Fmeasure()
                WFM = Measure.WeightedFmeasure()
                SM = Measure.Smeasure()
                EM = Measure.Emeasure()
                MAE = Measure.MAE()
                with torch.no_grad():
                    st = time.time()
                    for data in TestDataloader:
                        # image, mask, name, shape = data['img'].cuda(args.rank), data['label'], data['name'], data['shape']
                        image = data["image"].cuda()
                        mask = data["gt"]
                        test_text_feat = encode_text_batch(
                            tokenized_batch={
                                "input_ids": data["input_ids"],
                                "attention_mask": data["attention_mask"]
                            },
                            text_encoder=text_encoder
                        )
                        _, out, _ = net(image, test_text_feat)
                        out = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
                        pred = out.sigmoid().data.cpu().numpy().squeeze()
                        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)
                        mask = mask.numpy().astype(np.float32).squeeze()
                        mask /= (mask.max() + 1e-8)
                        FM.step(pred=pred * 255, gt=mask * 255)
                        WFM.step(pred=pred * 255, gt=mask * 255)
                        SM.step(pred=pred * 255, gt=mask * 255)
                        EM.step(pred=pred * 255, gt=mask * 255)
                        MAE.step(pred=pred * 255, gt=mask * 255)
                        fm = FM.get_results()["fm"]
                        wfm = WFM.get_results()["wfm"]
                        sm = SM.get_results()["sm"]
                        em = EM.get_results()["em"]
                        mae = MAE.get_results()["mae"]

                    results = {
                        "Smeasure": sm,
                        "wFmeasure": wfm,
                        "adpFm": fm["adp"],
                        "meanEm": em["curve"].mean(),
                        "MAE": mae,
                        "adpEm": em["adp"],
                        "maxEm": em["curve"].max(),
                        "meanFm": fm["curve"].mean(),
                        "maxFm": fm["curve"].max(),
                    }
                    print(results)
                    logging.info(
                        '{}{}'.format(TestDataloader.dataset.data_name, results))
                    print('dataset: {}, spend_time:{}'.format(
                        TestDataloader.dataset.data_name, (time.time() - st)))
                    maes[i] = mae
                net.train(True)
                # if TestDataloader.dataset.data_name == 'CAMO' and mae > 0.063:
                #     break
            mae_totel = maes[0] + maes[1] + maes[2]
            writer.add_scalar('MAE/CAMO', torch.tensor(maes[0]), global_step=epoch)
            writer.add_scalar('MAE/COD10K', torch.tensor(maes[1]), global_step=epoch)
            writer.add_scalar('MAE/NC4K', torch.tensor(maes[2]), global_step=epoch)
            if mae_totel < best_mae:
                best_mae = mae_totel
                best_epoch = epoch
                torch.save({
                    'state_dict': net.state_dict(),
                    'epoch': epoch
                }, args.save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
            print('Epoch:{}, MAE:{}, bestMAE:{}, bestEpoch:{}.'.format(epoch, mae_totel, best_mae, best_epoch))
            logging.info(
                '[Val Info]:Epoch:{} , MAE: {}, bestEpoch:{} bestMAE:{}'.format(epoch, mae_totel, best_epoch, best_mae))
        # if args.rank == 0 and (curr_epoch + 1) % 10 == 0:
        #     # ckpt_save_root = "./checkpoint/model_checkpoint1"
        #     # if not os.path.exists(ckpt_save_root):
        #     #     os.mkdir(ckpt_save_root)
        #     torch.save(net.state_dict(),
        #                args.save_path + "model_{}_loss_{:.5f}.pth".format(curr_epoch, running_loss_m / count)
        #                )


if __name__ == '__main__':
    best_mae = 1
    best_epoch = 0
    args = parse_args()
    main(args)