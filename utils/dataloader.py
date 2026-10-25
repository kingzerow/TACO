import os
from PIL import Image
from PIL import ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
import torch
from PIL import Image, ImageEnhance
# random.seed(2021)
# torch.manual_seed(2021)
# seed = 3407
# os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(3407)
# torch.cuda.manual_seed(3407)
# torch.cuda.manual_seed_all(3407)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)
def randomCrop(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)
def randomRotation(image, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, edge
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.kernel = np.ones((5, 5), np.uint8)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.ge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    def getFlip(self):
        p = random.randint(0, 1)
        self.flip = transforms.RandomHorizontalFlip(p)
        
    def __getitem__(self, index):
        # self.getFlip()
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        # edge = self.binary_loader(self.edges[index])

        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)
        # seed = np.random.randint(2147483647) # make a seed with numpy generator 
        # random.seed(2025) # apply this seed to img tranfsorms
        # torch.manual_seed(2025) # needed for torchvision 0.7
        # # data augumentation
        image, gt, edge = cv_random_flip(image, gt, edge)
        image, gt, edge = randomCrop(image, gt, edge)
        image, gt, edge = randomRotation(image, gt, edge)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        # image = self.flip(image)
        image = self.img_transform(image)
        # gt = self.flip(gt)
        gt = self.ge_transform(gt)
        # edge = self.flip(edge)		
        edge = self.edge_transform(edge)
        return image, gt, edge

    def filter_files(self):
        print(len(self.images), len(self.gts), len(self.edges))
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        edges = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            # print(img_path)
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            gt = Image.open(gt_path)
            gt = ImageOps.exif_transpose(gt)
            edge = Image.open(edge_path)
            edge = ImageOps.exif_transpose(edge)
            assert img.size == edge.size
            if img.size == gt.size and img.size == edge.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = ImageOps.exif_transpose(img)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = ImageOps.exif_transpose(img)
            # return img.convert('1')
            return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
    dataset = CamObjDataset(image_root, gt_root, edge_root, trainsize)
    print('dataset size:', len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        # self.index += 1
        # self.index = self.index % self.size
        # return image, gt, name
        self.index += 1
        return image, gt, name
    def load_data1(self):
        image = self.rgb_loader(self.images[self.index])
        image_for_post = self.images[self.index]
        image = self.transform(image).unsqueeze(0) # （1，3，704，704）
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name, image_for_post
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = ImageOps.exif_transpose(img)
            return img

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            img = ImageOps.exif_transpose(img)
            return img

    def __len__(self):
        return self.size
