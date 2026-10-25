import argparse
import os
import sys
import glob
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer (load prompt from JSON)")
    parser.add_argument("--version", default="models", type=str, help="pretrained model version")
    parser.add_argument("--vis_save_path", default="./infer_fixed/visualization", type=str, help="可视化结果保存路径")
    parser.add_argument("--mask_save_path", default="./infer_fixed/TextMask", type=str, help="二值分割掩码保存路径")
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="sam2", choices=["ori", "effi", "sam2"])
    parser.add_argument("--image_dir", type=str, default="../datasets/TrainDataset/Imgs", help="directory of images to process")
    # 新增参数：JSON描述文件路径
    # parser.add_argument("--prompt_json", type=str, default="./datasets/train.json", help="JSON file path containing image summary prompts")
    # parser.add_argument("--prompt_json", type=str, default="./datasets/train_label_size.json", help="JSON file path containing image summary prompts")
    # parser.add_argument("--prompt_json", type=str, default="/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train_with_caption_qwen.json", help="JSON file path containing image summary prompts")
    parser.add_argument("--prompt_json", type=str, default="/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train_with_caption_blip.json", help="JSON file path containing image summary prompts")
    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
    else:
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        x = (x - pixel_mean) / pixel_std
        resize_shape = None
    
    return x, resize_shape

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True,** kwargs
        )
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2 import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True,** kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model

def save_binary_mask(pred_mask, image_path, mask_save_path):
    """
    原生保存为PNG格式的二值掩码（编码+后缀均为PNG）
    """
    # 转换为二值图像：True→255（白色前景），False→0（黑色背景）
    binary_mask = pred_mask.astype(np.uint8) * 255
    # 提取原图名称（去掉后缀），使用.png后缀（编码与后缀一致）
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_file_path = os.path.join(mask_save_path, f"{img_name}.png")
    
    # 关键：指定PNG编码格式保存（cv2.IMWRITE_PNG_COMPRESSION控制压缩率）
    cv2.imwrite(
        mask_file_path, 
        binary_mask, 
        [int(cv2.IMWRITE_PNG_COMPRESSION), 0]  # 0=无压缩，适合掩码（无损）
    )
    return mask_file_path

def main(args):
    args = parse_args(args)
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 创建可视化和掩码保存目录
    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(args.mask_save_path, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 移除BLIP模型初始化（无需自动生成描述）
    tokenizer, model = init_models(args)

    # 核心改造1：读取JSON文件并构建{图片名: 文本描述}的字典映射
    with open(args.prompt_json, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    # 构建映射：key为image_filename，value为summary（文本描述）
    # img2prompt = {item['image_filename']: item['summary'] for item in prompt_data}
    # img2prompt = {item['image_filename']: item['summary_blip'] for item in prompt_data}
    img2prompt = {item['image_filename']: f"camouflaged object" for item in prompt_data}
    # img2prompt = {item['image_filename']: f"tiny camouflaged {item['label']},"+item['summary'] for item in prompt_data}

    # 获取文件夹内所有图片
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in directory: {args.image_dir}")
        return

    # 批量处理每张图片
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        # 提取图片basename（如COD10K-CAM-1-Aquatic-1-BatFish-1.jpg）
        img_basename = os.path.basename(image_path)
        
        # 核心改造2：匹配JSON中的文本描述，无匹配则跳过
        if img_basename not in img2prompt:
            print(f"Warning: No prompt found for {img_basename} in JSON, skip")
            continue
        prompt = img2prompt[img_basename]
        print(f"Loaded prompt from JSON: {prompt}")

        image_np = cv2.imread(image_path)
        if image_np is None:
            print(f"Failed to read {image_path}, skip")
            continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # 预处理（保留原有逻辑）
        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)
        image_sam, resize_shape = sam_preprocess(image_np, model_type=args.model_type)
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)

        # 编码prompt（保留原有逻辑）
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

        # 推理（保留原有逻辑）
        with torch.no_grad():
            pred_mask = model.inference(
                image_sam.unsqueeze(0),
                image_beit.unsqueeze(0),
                input_ids,
                resize_list=[resize_shape],
                original_size_list=original_size_list,
            )
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0  # 转为bool数组（True=前景，False=背景）

        # 保存可视化结果（保留原有逻辑）
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        vis_save_path = os.path.join(args.vis_save_path, f"{os.path.splitext(os.path.basename(image_path))[0]}_vis.png")
        cv2.imwrite(vis_save_path, save_img)

        # 保存二值分割掩码（保留原有逻辑）
        mask_path = save_binary_mask(pred_mask, image_path, args.mask_save_path)
        
        print(f"Visualization saved to {vis_save_path}")
        print(f"Binary mask saved to {mask_path}\n")

    print(f"Batch processing completed!")
    print(f"Visualization results: {args.vis_save_path}")
    print(f"Binary masks: {args.mask_save_path}")

if __name__ == "__main__":
    main(sys.argv[1:])