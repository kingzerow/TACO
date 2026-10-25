import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/user1/.cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import json
import pickle
from PIL import Image
from PIL import ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import random
import torch
from PIL import Image, ImageEnhance


# --------------------------- 文本工具函数（仅做Tokenize，不加载模型） ---------------------------
def load_tokenizer(encoder_type="mamba-130m"):
    """仅加载Tokenizer（不加载模型），避免在Dataset中初始化GPU模型"""
    if encoder_type == "clip-base":
        from transformers import CLIPTokenizerFast
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    elif encoder_type == 'clip-large':
        from transformers import CLIPTokenizerFast
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336")
    elif encoder_type == "bert-base":
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif encoder_type == "roberta-base":
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif encoder_type == "mamba-130m":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    else:
        raise NotImplementedError(f"不支持的文本编码器: {encoder_type}")
    return tokenizer


def tokenize_text(text: str, tokenizer, max_len=64):
    """仅做文本Tokenize，返回CPU张量（不执行模型推理）"""
    tokenized = tokenizer(
        text,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt"
    )
    # 返回CPU张量并移除batch维度，方便后续拼接
    return {
        "input_ids": tokenized["input_ids"].squeeze(0).cpu(),
        "attention_mask": tokenized["attention_mask"].squeeze(0).cpu()
    }


def load_text_encoder(encoder_type="mamba-130m", device="cuda"):
    """在主进程加载文本编码器（用于推理）"""
    if encoder_type == "clip-base":
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    elif encoder_type == 'clip-large':
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
    elif encoder_type == "bert-base":
        from transformers import BertModel
        text_encoder = BertModel.from_pretrained("bert-base-uncased")
    elif encoder_type == "roberta-base":
        from transformers import RobertaModel
        text_encoder = RobertaModel.from_pretrained("roberta-base")
    elif encoder_type == "mamba-130m":
        from transformers import MambaModel
        text_encoder = MambaModel.from_pretrained("state-spaces/mamba-130m-hf")
    else:
        raise NotImplementedError(f"不支持的文本编码器: {encoder_type}")

    # 冻结参数，移到指定设备
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    text_encoder.eval()
    text_encoder = text_encoder.to(device)
    return text_encoder


def encode_text_batch(tokenized_batch, text_encoder, device="cuda"):
    """主进程中批量编码文本（GPU推理）"""
    # 将batch数据移到GPU
    input_ids = tokenized_batch["input_ids"].to(device)
    attention_mask = tokenized_batch["attention_mask"].to(device)
    
    with torch.no_grad():
        if "clip" in str(text_encoder.__class__.__name__).lower():
            # CLIP的forward参数是input_ids和attention_mask
            encoded = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        elif "mamba" in str(text_encoder.__class__.__name__).lower():
            # Mamba仅需要input_ids
            encoded = text_encoder(input_ids=input_ids)
        else:
            # BERT/RoBERTa
            encoded = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    
    # 整理输出特征
    # output = {
    #     "text_features": encoded.last_hidden_state,  # [batch, max_len, dim]
    #     "text_attention_mask": attention_mask.ne(1).bool()  # mask
    # }
    output = {
        "text_features": encoded.last_hidden_state,  # [batch, max_len, dim]
        "text_attention_mask": attention_mask.bool()  # mask
    }
    if hasattr(encoded, "pooler_output"):
        output["text_pooler"] = encoded.pooler_output  # [batch, dim]
    return output


# --------------------------- 原有数据增强函数（保留） ---------------------------
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
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1, (image_height + crop_win_height) >> 1
    )
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, edge


# --------------------------- 训练数据集类（仅做CPU操作） ---------------------------
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, text_json_path, key, trainsize,
                 text_encoder_type="mamba-130m", max_text_len=64, use_text_cache=True,
                 cache_dir="./text_cache/clips"):
        # 基础参数
        self.trainsize = trainsize
        self.max_text_len = max_text_len

        # 加载图像/标签/边缘路径
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                    if f.endswith('.png')]
        self.edges = [os.path.join(edge_root, f) for f in os.listdir(edge_root)
                      if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)

        # 加载文本映射和Tokenizer（仅CPU）
        self.img2summary = self.load_text_json(text_json_path, key)
        self.tokenizer = load_tokenizer(text_encoder_type)
        self.filter_files()  # 过滤无效文件
        self.size = len(self.images)
        self.kernel = np.ones((5, 5), np.uint8)

        # 文本缓存（仅缓存Tokenize结果）
        self.use_text_cache = use_text_cache
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.text_cache = self.load_text_cache() if use_text_cache else {}

        # 图像变换（原有逻辑）
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.ge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def load_text_json(self, json_path, key):
        """加载文本映射"""
        print(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        img2summary = {}
        for item in data_list:
            img_filename = item["image_filename"]
            summary = item["summary"]
            if key == "class_prompt":
                label = item["label"]
                class_prompt = f"A photo of a camouflaged {label}"
                img2summary[img_filename] = class_prompt # class_prompt
            elif key == "summary":
                img2summary[img_filename] = summary # class_prompt
            elif key == "summary_blip":
                img2summary[img_filename] = item['summary_blip'] # class_prompt
            elif key == "summary_qwen":
                # img2summary[img_filename] = item['summary_qwen'] # class_prompt
                img2summary[img_filename] = item.get("summary_qwen", item.get("summary"))
            elif key == "class_and_summary":
                img2summary[img_filename] = f"small camouflaged {item['label']},"+summary
            elif key == "other":
                img2summary[img_filename] = "camouflaged object"
        return img2summary

    def load_text_cache(self):
        """加载Tokenize缓存"""
        cache_path = os.path.join(self.cache_dir, "text_token_cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_text_cache(self):
        """保存Tokenize缓存"""
        cache_path = os.path.join(self.cache_dir, "text_token_cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(self.text_cache, f)

    def get_text_tokens(self, img_filename):
        """获取文本Tokenize结果（优先缓存）"""
        if img_filename in self.text_cache:
            return self.text_cache[img_filename]
        
        if img_filename not in self.img2summary:
            raise ValueError(f"图像 {img_filename} 无对应文本")
        summary = self.img2summary[img_filename]
        text_tokens = tokenize_text(
            summary, self.tokenizer, max_len=self.max_text_len
        )

        if self.use_text_cache:
            self.text_cache[img_filename] = text_tokens
            self.save_text_cache()
        return text_tokens

    def filter_files(self):
        """过滤无效文件"""
        print(f"过滤前 - 图像数: {len(self.images)}, 标签数: {len(self.gts)}, 边缘数: {len(self.edges)}")
        images, gts, edges = [], [], []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            # 尺寸检查
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            gt = Image.open(gt_path)
            gt = ImageOps.exif_transpose(gt)
            edge = Image.open(edge_path)
            edge = ImageOps.exif_transpose(edge)
            if img.size != gt.size or img.size != edge.size:
                continue
            # 文本检查
            img_filename = os.path.basename(img_path)
            if img_filename not in self.img2summary:
                continue
            images.append(img_path)
            gts.append(gt_path)
            edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges
        print(f"过滤后 - 图像数: {len(self.images)}, 标签数: {len(self.gts)}, 边缘数: {len(self.edges)}")

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return ImageOps.exif_transpose(img)

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).convert('L')
            return ImageOps.exif_transpose(img)

    def __getitem__(self, index):
        # 加载图像/标签/边缘
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)

        # 数据增强
        image, gt, edge = cv_random_flip(image, gt, edge)
        image, gt, edge = randomCrop(image, gt, edge)
        image, gt, edge = randomRotation(image, gt, edge)
        image = colorEnhance(image)
        gt = randomPeper(gt)

        # 图像变换
        image = self.img_transform(image)
        gt = self.ge_transform(gt)
        edge = self.edge_transform(edge)

        # 加载文本Token（仅CPU）
        img_filename = os.path.basename(self.images[index])
        text_tokens = self.get_text_tokens(img_filename)

        return {
            "image": image,
            "gt": gt,
            "edge": edge,
            "input_ids": text_tokens["input_ids"],
            "attention_mask": text_tokens["attention_mask"],
            "img_filename": img_filename
        }

    def __len__(self):
        return self.size


# --------------------------- 测试数据集类（仅做CPU操作） ---------------------------
# --------------------------- 修改后的测试数据集类（实现标准Dataset接口） ---------------------------
class test_dataset(data.Dataset):  # 继承data.Dataset，符合PyTorch标准
    def __init__(self, image_root, gt_root, text_json_path, key, testsize,
                 text_encoder_type="mamba-130m", max_text_len=64):
        self.testsize = testsize
        self.max_text_len = max_text_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_name = image_root.split('/')[-1]
        # 加载图像/标签路径
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root)
                    if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        # 加载文本JSON
        self.img2summary = self.load_text_json(text_json_path, key)

        # 文本编码初始化：仅加载Tokenizer（CPU），模型移到主进程
        self.tokenizer = load_tokenizer(text_encoder_type)  # 用你之前重构的load_tokenizer

        # 图像变换（原有逻辑）
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)  # 与__len__对应

    def load_text_json(self, json_path, key):
        """同训练集的文本加载逻辑"""
        # json_path = './datasets/dataset/test_with_label.json'
        with open(json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        img2summary = {}
        for item in data_list:
            img_filename = item["image_filename"]
            # summary = item["summary"]
            if key == "class_prompt":
                label = item["label"]
                img2summary[img_filename] = f"A photo of a camouflaged {label}" # class_prompt
                pass
            elif key == "summary":
                img2summary[img_filename] = item["summary"]
            elif key == "summary_blip":
                img2summary[img_filename] = item["summary_blip"] # class_prompt
            elif key == "summary_qwen":
                img2summary[img_filename] = item['summary_qwen'] # class_prompt
            elif key == "class_and_summary":
                label = item["label"]
                img2summary[img_filename] = f"small camouflaged {label},"+item['summary']
            elif key == "class_and_summary1":
                label = item["label"]
                img2summary[img_filename] = f"camouflaged {label},"+item['summary']
            elif key == "other":
                # img2summary[img_filename] = f"A photo of a camouflaged object"
                img2summary[img_filename] = f"camouflaged object"
        return img2summary

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
    
    # 核心：实现PyTorch要求的__getitem__方法，DataLoader会调用这个方法通过idx获取数据
    def __getitem__(self, idx):
        # 根据索引加载单张图像/标签
        image = self.rgb_loader(self.images[idx])
        gt = self.binary_loader(self.gts[idx])
        img_filename = os.path.basename(self.images[idx])
        # 处理文件名（原有逻辑）
        name = img_filename.split('.jpg')[0] + '.png' if img_filename.endswith('.jpg') else img_filename
        image_for_post = self.images[idx]  # 保留原图路径，用于后续后处理

        # 图像变换（原有逻辑）
        image = self.transform(image)  # 无需unsqueeze(0)，DataLoader会自动拼接batch
        # 核心修改：将PIL格式的GT转为张量（使用你定义的gt_transform）
        gt = self.gt_transform(gt)  # 新增这一行，直接转张量
        # 仅做文本Tokenize（CPU，原有逻辑），不执行模型推理
        text_tokens = tokenize_text(  # 用你之前重构的tokenize_text
            self.img2summary[img_filename], self.tokenizer, max_len=self.max_text_len
        )

        return {
            "image": image,               # [3, testsize, testsize]
            "gt": gt,                     # 原始PIL图像，按需在主进程转换
            "name": name,                 # 图像名称
            "image_for_post": image_for_post,  # 原图路径
            "input_ids": text_tokens["input_ids"],  # [max_len]
            "attention_mask": text_tokens["attention_mask"]  # [max_len]
        }
    
    # 核心：实现PyTorch要求的__len__方法，返回数据集总长度
    def __len__(self):
        return self.size

    # 保留你原有load_data/load_data1方法（兼容你之前的推理代码，可选）
    def load_data(self):
        data = self.__getitem__(self.index)
        # 为了兼容原有逻辑，给image增加batch维度
        data["image"] = data["image"].unsqueeze(0)
        data["input_ids"] = data["input_ids"].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"].unsqueeze(0)
        self.index += 1
        return data

    def load_data1(self):
        data = self.__getitem__(self.index)
        data["image"] = data["image"].unsqueeze(0)
        data["input_ids"] = data["input_ids"].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"].unsqueeze(0)
        self.index += 1
        return data


# --------------------------- 数据加载器封装 ---------------------------
def get_loader(image_root, gt_root, edge_root, text_json_path, key, batchsize, trainsize,
               shuffle=True, num_workers=8, pin_memory=True, text_encoder_type="mamba-130m", cache_dir='./text_cache'):
    dataset = CamObjDataset(
        image_root=image_root,
        gt_root=gt_root,
        edge_root=edge_root,
        text_json_path=text_json_path,
        key=key,
        trainsize=trainsize,
        text_encoder_type=text_encoder_type,
        cache_dir=cache_dir
    )
    print('dataset size:', len(dataset))
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: {
            "image": torch.stack([item["image"] for item in x]),
            "gt": torch.stack([item["gt"] for item in x]),
            "edge": torch.stack([item["edge"] for item in x]),
            "input_ids": torch.stack([item["input_ids"] for item in x]),
            "attention_mask": torch.stack([item["attention_mask"] for item in x]),
            "img_filename": [item["img_filename"] for item in x]
        }
    )
    return data_loader


# --------------------------- 使用示例（主进程） ---------------------------
if __name__ == "__main__":
    # 1. 主进程初始化设备和文本编码器（GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder_type = "mamba-130m"  # 可选 "clip-base", "bert-base", "roberta-base", "mamba-130m"
    text_encoder = load_text_encoder(text_encoder_type, device=device)
    train_path = "../datasets/TrainDataset"
    # 2. 创建训练加载器（num_workers可正常设置）
    train_loader = get_loader(
        image_root='{}/Imgs/'.format(train_path),
        gt_root='{}/K=10_P=5/GT/'.format(train_path),
        edge_root='{}/K=10_P=5/Edge/'.format(train_path),
        text_json_path='datasets/train.json',  # 你的文本JSON路径 
        batchsize=2,
        trainsize=512,
        text_encoder_type=text_encoder_type,
        num_workers=4  # 可正常使用多进程
    )

    # 3. 迭代数据并在主进程编码文本
    for batch in train_loader:
        print("图像形状:", batch["image"].shape)  # [2, 3, 512, 512]
        print("Input_ids形状:", batch["input_ids"].shape)  # [2, 64]
        
        # 主进程中执行文本编码（GPU）
        text_feat = encode_text_batch(
            tokenized_batch={
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"]
            },
            text_encoder=text_encoder,
            device=device
        )
        print("文本特征形状:", text_feat["text_features"].shape)  # [2, 64, 512]
        break

    # 4. 测试集示例
    test_path = "../datasets/TestDataset/CAMO"
    test_set = test_dataset(
        image_root='{}/Imgs/'.format(test_path),
        gt_root='{}/GT/'.format(test_path),
        text_json_path='datasets/test.json',
        testsize=512,
        text_encoder_type=text_encoder_type
    )
    test_data = test_set.load_data()
    print("测试集Input_ids形状:", test_data["input_ids"].shape)  # [1, 64]
    
    # 测试集文本编码
    test_text_feat = encode_text_batch(
        tokenized_batch={
            "input_ids": test_data["input_ids"],
            "attention_mask": test_data["attention_mask"]
        },
        text_encoder=text_encoder,
        device=device
    )
    print("测试集文本特征形状:", test_text_feat["text_features"].shape)  # [1, 64, 512]