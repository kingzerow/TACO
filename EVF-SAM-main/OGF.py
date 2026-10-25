import os
import time
import cv2
import numpy as np
# 原理是：点生成的 mask 更可信定位，文本生成的 mask 用来补充目标区域。
POINT_DIR = "../datasets/TrainDataset/mix2/GT"
TEXT_DIR = "./infer_ref/TextMask"
# OUT_DIR = "../datasets/TrainDataset/mix2_fixed/GT"

MIN_OVERLAP = 0.3
DILATE_ITER = 3
TH = 127


def file_map(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return {
        os.path.splitext(f)[0]: f
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    }


def fuse(mp, mt):
    mp = mp > TH
    mt = mt > TH

    if mp.sum() == 0:
        return mt.astype(np.uint8) * 255
    if mt.sum() == 0:
        return mp.astype(np.uint8) * 255

    inter = mp & mt
    overlap = inter.sum() / (mp.sum() + 1e-6)

    if overlap < MIN_OVERLAP:
        out = mp
    else:
        kernel = np.ones((5, 5), np.uint8)
        mp_d = cv2.dilate(mp.astype(np.uint8), kernel, iterations=DILATE_ITER).astype(bool)
        out = mp | (mt & mp_d)

    return out.astype(np.uint8) * 255


def main():
    # os.makedirs(OUT_DIR, exist_ok=True)

    p_files = file_map(POINT_DIR)
    t_files = file_map(TEXT_DIR)

    ok, fail = 0, []
    image_times = []
    total_start = time.perf_counter()

    for name, pf in p_files.items():
        if name not in t_files:
            print(f"[MISS TEXT] {name}")
            fail.append(name)
            continue


        image_start = time.perf_counter()
        mp = cv2.imread(os.path.join(POINT_DIR, pf), 0)
        mt = cv2.imread(os.path.join(TEXT_DIR, t_files[name]), 0)
        if mp is None or mt is None:
            raise RuntimeError("read mask failed")
        if mp.shape != mt.shape:
            mt = cv2.resize(mt, (mp.shape[1], mp.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = fuse(mp, mt)
        # saved = cv2.imwrite(os.path.join(OUT_DIR, name + ".png"), out)
        # if not saved:
        #     raise RuntimeError("write fused mask failed")
        image_time = time.perf_counter() - image_start
        image_times.append(image_time)
        ok += 1
        if ok % 100 == 0:
            print(
                f"processed {ok}, last: {image_time:.6f} s/image, "
                f"mean: {np.mean(image_times):.6f} s/image"
            )

        # print(f"[FAIL] {name}: {e}")
        # fail.append(name)

    total_time = time.perf_counter() - total_start
    print("\n========== OGF Fusion Runtime ==========")
    print("success:", ok)
    print("failed:", len(fail))
    print(f"total wall time: {total_time:.6f} s")
    if image_times:
        print(f"mean runtime  : {np.mean(image_times):.6f} s/image")
        print(f"std. runtime  : {np.std(image_times):.6f} s/image")
        print(f"median runtime: {np.median(image_times):.6f} s/image")

    # if fail:
    #     with open(os.path.join(OUT_DIR, "failed.txt"), "w") as f:
    #         f.write("\n".join(fail))


if __name__ == "__main__":
    main()
