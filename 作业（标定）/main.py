import os
import sys
import numpy as np
import cv2 as cv
import glob as gb

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
PARAM_FILE = os.path.join(BASE, "标定参数.npz")
OUT_DIR = os.path.join(BASE, "去畸变结果")
CMP_FILE = os.path.join(BASE, "去畸变对比.jpg")
SIMPLE_FILE = os.path.join(BASE, "去畸变_简单版.jpg")

PREVIEW = "--no-preview" not in sys.argv


def read_image(path):
    return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)


def save_image(path, img, quality=95):
    cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])[1].tofile(path)


def load_params(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "找不到 %s\n请先完整运行一次 lamb.py（一路按到看完，别按空格），"
            "它会把 K、D 存成这个文件。" % path)

    d = np.load(path)
    mtx = d["mtx"]
    dist = d["dist"].reshape(-1)
    size = (int(d["width"]), int(d["height"]))
    return mtx, dist, size


def undistort_simple(img, mtx, dist):
    return cv.undistort(img, mtx, dist)


def build_maps(mtx, dist, size, alpha):
    new_K, roi = cv.getOptimalNewCameraMatrix(mtx, dist, size, alpha, size)
    map1, map2 = cv.initUndistortRectifyMap(
        mtx, dist, None, new_K, size, cv.CV_32FC1)
    return map1, map2, new_K, roi


def apply_maps(img, map1, map2):
    return cv.remap(img, map1, map2, cv.INTER_LINEAR)


def make_comparison(panels, labels, tile_w=560):
    bar_h = 40
    tiles = []
    for img, label in zip(panels, labels):
        h, w = img.shape[:2]
        tile = cv.resize(img, (tile_w, int(h * tile_w / w)))
        cv.rectangle(tile, (0, 0), (tile.shape[1], bar_h), (0, 0, 0), -1)
        cv.putText(tile, label, (10, 28), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2, cv.LINE_AA)
        tiles.append(tile)
    return cv.hconcat(tiles)


def main():
    mtx, dist, size = load_params(PARAM_FILE)
    print("读取标定参数完成")
    print("图像尺寸 (宽, 高):", size)
    print("内参矩阵 K:\n", mtx)
    print("畸变系数 D = [k1, k2, p1, p2, k3]:\n", dist)

    os.makedirs(OUT_DIR, exist_ok=True)

    map0_x, map0_y, new_K0, roi0 = build_maps(mtx, dist, size, alpha=0)
    map1_x, map1_y, new_K1, roi1 = build_maps(mtx, dist, size, alpha=1)

    print("alpha=0 的新内参主点: cx=%.2f, cy=%.2f" % (new_K0[0, 2], new_K0[1, 2]))
    print("alpha=0 有效区域 roi (x,y,w,h):", roi0)
    print("alpha=1 有效区域 roi (x,y,w,h):", roi1,
          "  <- alpha=1 时按这个矩形裁，就能把黑边去掉")

    images = sorted(gb.glob(os.path.join(IMG_DIR, "*.jpg")))
    print("\n待处理图片:", len(images), "张")

    example = None
    for fname in images:
        img = read_image(fname)
        if img is None:
            print("读取失败，跳过:", os.path.basename(fname))
            continue

        if (img.shape[1], img.shape[0]) != size:
            print("尺寸不符，跳过:", os.path.basename(fname), img.shape[1::-1])
            continue

        fixed = apply_maps(img, map0_x, map0_y)
        save_image(os.path.join(OUT_DIR, os.path.basename(fname)), fixed)

        if example is None:
            example = {
                "orig": img,
                "simple": undistort_simple(img, mtx, dist),
                "a0": fixed,
                "a1": apply_maps(img, map1_x, map1_y),
            }

    print("批量结果已存到:", OUT_DIR)

    if example is not None:
        save_image(SIMPLE_FILE, example["simple"])
        print("方法 A 示例已存:", SIMPLE_FILE)

        comparison = make_comparison(
            [example["orig"], example["simple"], example["a0"], example["a1"]],
            ["1 Original (distorted)", "2 cv.undistort", "3 alpha=0", "4 alpha=1"])
        save_image(CMP_FILE, comparison)
        print("四栏对比图已存:", CMP_FILE)

        x, y, w, h = roi1
        cropped = example["a1"][y:y + h, x:x + w]
        print("alpha=1 按 roi 裁剪后的尺寸:", cropped.shape[1::-1],
              "（原始尺寸:", size, "）")

        if PREVIEW:
            shown = cv.resize(comparison, dsize=None, fx=0.75, fy=0.75)
            cv.imshow("去畸变对比   [任意键关闭]", shown)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
