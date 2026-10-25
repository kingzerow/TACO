import os, json, base64
from tqdm import tqdm
from openai import OpenAI

# ======================
# config
# ======================
IMAGE_DIR = "../datasets/TrainDataset/Imgs"
INPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train.json"
OUTPUT_JSON = "/home/user1/wrs/FCT-SAM-WSCOD-main/datasets/train_with_caption.json"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BATCH_SIZE = 8

prompt = """
Describe the camouflaged object in this image.
Focus on object type, appearance, and camouflage strategy.
"""

# ======================
# utils
# ======================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_gpt(image_path):
    img_b64 = encode_image(image_path)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=100
    )

    return resp.choices[0].message.content


# ======================
# load json
# ======================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

image_map = {item["image_filename"]: item for item in data}
files = list(image_map.keys())

# ======================
# batch loop (logical batch)
# ======================
for i in tqdm(range(0, len(files), BATCH_SIZE)):
    batch = files[i:i + BATCH_SIZE]

    for f in batch:
        path = os.path.join(IMAGE_DIR, f)

        try:
            caption = call_gpt(path)
            image_map[f]["summary"] = caption

        except Exception as e:
            image_map[f]["summary"] = f"ERROR: {str(e)}"

# ======================
# write back
# ======================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(list(image_map.values()), f, indent=2, ensure_ascii=False)

print("Saved:", OUTPUT_JSON)