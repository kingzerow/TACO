import os, json, glob
from PIL import Image
from tqdm import tqdm
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from transformers import BlipProcessor, BlipForConditionalGeneration


# IMAGE_DIR = "../datasets/TrainDataset/Imgs"
# INPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train.json"
# OUTPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train_with_caption_blip.json"
IMAGE_DIR = "../datasets/TestDataset/COD10K/Imgs"
INPUT_JSON = "/home/user1/wrs/datasets/TestDataset/COD10K/test_with_label.json"
OUTPUT_JSON = "/home/user1/wrs/datasets/TestDataset/COD10K/test_with_label_blip.json"
BATCH_SIZE = 32

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).cuda()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model.eval()

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

record_dict = {item["image_filename"]: item for item in data}

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

image_paths.sort()

def generate_batch_caption(batch_paths):
    images = [Image.open(p).convert("RGB") for p in batch_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.module.generate(**inputs, max_new_tokens=50) if isinstance(model, torch.nn.DataParallel) else model.generate(**inputs, max_new_tokens=50)

    return processor.batch_decode(outputs, skip_special_tokens=True)

for start in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[start:start + BATCH_SIZE]

    try:
        captions = generate_batch_caption(batch_paths)

        for path, caption in zip(batch_paths, captions):
            filename = os.path.basename(path)
            if filename in record_dict:
                record_dict[filename]["summary_blip"] = caption.strip()

    except Exception as e:
        print(f"Batch {start} failed: {e}")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_JSON}")