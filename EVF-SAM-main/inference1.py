import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig, BlipProcessor, BlipForConditionalGeneration
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer")
    parser.add_argument("--version", default="models", type=str, help="pretrained model version")
    parser.add_argument("--vis_save_path", default="./COD", type=str)
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
    # 改动1：将单张图片路径改为文件夹路径（批量处理）
    parser.add_argument("--image_dir", type=str, default="../datasets/TrainDataset/Imgs", help="directory of images to process")
    # 移除手动prompt参数，改为自动生成
    return parser.parse_args(args)


def sam_preprocess(
    x: np.ndarray,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
    model_type="ori") -> torch.Tensor:
    assert img_size==1024, \
        "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
    
    # Normalize colors
    if model_type=="ori":
        x = ResizeLongestSide(img_size).apply_image(x)
        h, w = resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = (x - pixel_mean) / pixel_std
        # Pad
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

# 新增：初始化BLIP模型用于生成文本描述
def init_blip_model(device):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    vlm_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    return processor, vlm_model

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

# 新增：生成图像描述的函数（复用第二段代码逻辑）
def generate_image_caption(image_np, processor, vlm_model, device):
    # 将numpy图像转为PIL图像（适配BLIP输入）
    image_pil = Image.fromarray(image_np)
    # 生成图像描述
    vlm_inputs = processor(image_pil, return_tensors="pt").to(device)
    vlm_outputs = vlm_model.generate(**vlm_inputs, output_hidden_states=True)
    description = processor.decode(vlm_outputs[0], skip_special_tokens=True)
    return description

def main(args):
    args = parse_args(args)
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 改动2：创建批量保存目录
    os.makedirs(args.vis_save_path, exist_ok=True)
    
    # 初始化BLIP模型（用于生成文本描述）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_processor, blip_model = init_blip_model(device)
    blip_model.eval()

    # 初始化EVF-SAM模型和tokenizer
    tokenizer, model = init_models(args)

    # 改动3：获取文件夹内所有图片（支持jpg/png）
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
        
        # 读取图片
        image_np = cv2.imread(image_path)
        if image_np is None:
            print(f"Failed to read {image_path}, skip")
            continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # 预处理
        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device)
        image_sam, resize_shape = sam_preprocess(image_np, model_type=args.model_type)
        image_sam = image_sam.to(dtype=model.dtype, device=model.device)

        # 改动4：自动生成图片描述（替换手动prompt）
        with torch.no_grad():
            prompt = generate_image_caption(image_np, blip_processor, blip_model, device)
        print(f"Generated prompt for {os.path.basename(image_path)}: {prompt}")

        # 编码prompt
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)

        # 推理
        with torch.no_grad():
            pred_mask = model.inference(
                image_sam.unsqueeze(0),
                image_beit.unsqueeze(0),
                input_ids,
                resize_list=[resize_shape],
                original_size_list=original_size_list,
            )
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        # 保存可视化结果
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        
        # 生成保存路径
        img_basename = os.path.basename(image_path)
        save_path = os.path.join(args.vis_save_path, f"{os.path.splitext(img_basename)[0]}_vis.png")
        cv2.imwrite(save_path, save_img)
        print(f"Result saved to {save_path}\n")

    print(f"Batch processing completed! All results saved to {args.vis_save_path}")

if __name__ == "__main__":
    main(sys.argv[1:])