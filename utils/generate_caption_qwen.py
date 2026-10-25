import os, json, time
from PIL import Image
from tqdm import tqdm
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# IMAGE_DIR = "../datasets/TrainDataset/Imgs"
# INPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train.json"
# OUTPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train_with_caption_qwen1.json"
IMAGE_DIR = "../datasets/TestDataset/COD10K/Imgs"
INPUT_JSON = "/home/user1/wrs/datasets/TestDataset/COD10K/test_with_label.json"
OUTPUT_JSON = "/home/user1/wrs/datasets/TestDataset/COD10K/test_with_label_qwen.json"
MODEL = "Qwen/Qwen2-VL-7B-Instruct"
BATCH = 1
PROMPT = "Describe the camouflaged object concisely."
RESIZE_SIZE = (416, 416)
MAX_IMAGES = 5000

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

with open(INPUT_JSON) as f:
    data = json.load(f)

mp = {x["image_filename"]: x for x in data}
files = list(mp.keys())
if MAX_IMAGES is not None:
    files = files[:MAX_IMAGES]
print(f"Running Qwen captioning on {len(files)} images "
      f"({'all' if MAX_IMAGES is None else 'timing subset'}).")


def build_texts(n):
    msg = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]
    }]
    return [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for _ in range(n)]


@torch.no_grad()
def infer(imgs):
    inputs = processor(text=build_texts(len(imgs)), images=imgs, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(**inputs, max_new_tokens=60)

    out = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(out, skip_special_tokens=True)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


total_images = 0
total_infer_time = 0.0


for i in tqdm(range(0, len(files), BATCH)):
    batch = files[i:i+BATCH]

    imgs, keep = [], []
    for f in batch:
        p = os.path.join(IMAGE_DIR, f)
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            if RESIZE_SIZE is not None:
                img = img.resize(RESIZE_SIZE, Image.BICUBIC)
            imgs.append(img)
            keep.append(f)

    if not imgs:
        continue

    try:
        sync_cuda()
        start = time.perf_counter()
        caps = infer(imgs)
        sync_cuda()
        elapsed = time.perf_counter() - start
        total_infer_time += elapsed
        total_images += len(imgs)
        for f, c in zip(keep, caps):
            mp[f]["summary_qwen"] = c.strip()
    except Exception as e:
        print("ERR:", i, e)

with open(OUTPUT_JSON, "w") as f:
    json.dump(list(mp.values()), f, indent=2, ensure_ascii=False)

print("DONE:", OUTPUT_JSON)
if total_images:
    resize_text = "original" if RESIZE_SIZE is None else f"{RESIZE_SIZE[0]}x{RESIZE_SIZE[1]}"
    print(f"Qwen caption inference: {total_infer_time:.2f}s total, "
          f"{total_infer_time / total_images:.4f}s/img, "
          f"batch={BATCH}, resize={resize_text}")
