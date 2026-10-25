import os, json, glob, time
from PIL import Image
from tqdm import tqdm
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from transformers import BlipProcessor, BlipForConditionalGeneration


IMAGE_DIR = "../datasets/TrainDataset/Imgs"
INPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train.json"
OUTPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train_with_caption_blip1.json"
# IMAGE_DIR = "../datasets/TestDataset/CAMO/Imgs"
# INPUT_JSON = "/home/user1/wrs/datasets/TestDataset/CAMO/test_with_label.json"
# OUTPUT_JSON = "/home/user1/wrs/datasets/TestDataset/CAMO/test_with_label.json"
BATCH_SIZE = 1
RESIZE_SIZE = (416, 416)
MAX_IMAGES = 32

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
image_paths = [p for p in image_paths if os.path.basename(p) in record_dict]
if MAX_IMAGES is not None:
    image_paths = image_paths[:MAX_IMAGES]
print(f"Running BLIP captioning on {len(image_paths)} images "
      f"({'all' if MAX_IMAGES is None else 'timing subset'}).")

def generate_batch_caption(batch_paths):
    images = []
    for p in batch_paths:
        img = Image.open(p).convert("RGB")
        if RESIZE_SIZE is not None:
            img = img.resize(RESIZE_SIZE, Image.BICUBIC)
        images.append(img)
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.module.generate(**inputs, max_new_tokens=50) if isinstance(model, torch.nn.DataParallel) else model.generate(**inputs, max_new_tokens=50)

    return processor.batch_decode(outputs, skip_special_tokens=True)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


total_images = 0
total_infer_time = 0.0

for start in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[start:start + BATCH_SIZE]

    try:
        sync_cuda()
        t0 = time.perf_counter()
        captions = generate_batch_caption(batch_paths)
        sync_cuda()
        elapsed = time.perf_counter() - t0
        total_infer_time += elapsed
        total_images += len(batch_paths)

        for path, caption in zip(batch_paths, captions):
            filename = os.path.basename(path)
            if filename in record_dict:
                record_dict[filename]["summary_blip"] = caption.strip()

    except Exception as e:
        print(f"Batch {start} failed: {e}")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_JSON}")
if total_images:
    resize_text = "original" if RESIZE_SIZE is None else f"{RESIZE_SIZE[0]}x{RESIZE_SIZE[1]}"
    print(f"BLIP caption inference: {total_infer_time:.2f}s total, "
          f"{total_infer_time / total_images:.4f}s/img, "
          f"batch={BATCH_SIZE}, resize={resize_text}")
