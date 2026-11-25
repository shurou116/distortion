"""
====================================================================
 去除镜头畸变（图像矫正 / Undistortion）
====================================================================

这个文件只做一件事：用 lamb.py 标定出来的 K 和 D，把 images/ 里的
照片"掰直"，并把矫正前、矫正后的效果存盘对比。

运行：
    python main.py                 处理全部图片 + 弹窗看对比图
    python main.py --no-preview    只保存文件，不弹窗

前置条件：先完整跑一次 lamb.py（一路按到看完，让它保存结果），
         会生成 标定参数.npz。这个文件里装着 K 和 D。

--------------------------------------------------------------------
 【一】先搞懂：畸变到底是什么？
--------------------------------------------------------------------
理想相机是"针孔模型"：光线直着穿过一个小孔打到传感器上，
空间里的一条直线，拍出来一定还是一条直线。

但真实镜头是一组曲面玻璃。光线经过镜片边缘时会被"掰弯"，
于是照片里的直线会变成弧线。你拍的标定板照片，本来格子边缘
应该是笔直的，靠近画面边角的地方就会往外鼓或往里凹，这就是畸变。

畸变分两类，对应 D 里的 5 个数：

    D = [k1, k2, p1, p2, k3]

  (1) 径向畸变 k1, k2, k3
      —— 离画面中心越远越明显，是镜片曲面本身造成的。
      k1 > 0 一般是"桶形"：边缘往外鼓，直线变成向外凸的弧；
      k1 < 0 一般是"枕形"：边缘往里收，直线变成向内凹的弧。
      k3 是更高阶的修正项，通常用得少但少了它边角会拟合不准。

  (2) 切向畸变 p1, p2
      —— 镜头和传感器没装正（两者不平行），画面像被斜着拽了一下。
      这项一般很小，本次标定出来约 -0.00009 和 -0.00046，几乎可以忽略。

本次 19 张图标定得到：
    D = [0.097804, -0.549723, -0.000090, -0.000459, 0.726477]
k1 为正、k2 为负、k3 为正，是很常见的镜头组合，说明确实存在径向畸变。

--------------------------------------------------------------------
 【二】K 是什么？为什么去畸变必须要有它？
--------------------------------------------------------------------
    K = [ fx   0   cx ]        fx, fy：焦距，单位是"像素"
        [  0  fy   cy ]        cx, cy：主点，光轴打在传感器上的位置
        [  0   0    1 ]               一般接近图片中心

畸变公式是在"归一化相机坐标"里定义的（也就是把像素坐标减掉 cx/cy、
再除以 fx/fy 之后、没有量纲的坐标）。所以要把一张图矫正回来，
必须先知道 K，才能把像素 (u, v) 换算到那个坐标系里套公式，
算完再换回像素坐标。D 单独一个人是没法干活的，K 和 D 必须配对使用。

--------------------------------------------------------------------
 【三】cv.undistort 内部到底做了什么？（最容易想反的地方）
--------------------------------------------------------------------
很多人以为是"把原图上每个像素搬到新位置"，其实是反过来的：

  对【矫正后】图片上的每一个像素 (u_new, v_new)：
    ① 用 K 的逆，把它变回归一化相机坐标 (x, y)
    ② 套用畸变公式（用 D），算出"这个点在【有畸变的原图】上
       实际落在哪个位置" (u_old, v_old)
    ③ 去原图 (u_old, v_old) 处取颜色。这个坐标通常不是整数
       （比如 312.47, 88.91），所以用双线性插值从周围 4 个像素
       算出颜色，填进矫正图。

这叫"反向映射 / 后向映射"(backward mapping)。
好处：矫正图的每个格子都一定会被填满，不会出现空洞或锯齿状裂缝。
如果反过来做正向映射，有些格子会没人填、有些格子会被重复填。

--------------------------------------------------------------------
 【四】两种方法，什么时候用哪个？
--------------------------------------------------------------------
方法 A：cv.undistort(图, K, D)
    一行搞定，适合临时看一张图。
    不指定新内参时直接用原 K，图片尺寸不变，四个角常会出现
    黑色无效区（因为那些位置映射回原图时已经超出边界了）。

方法 B：getOptimalNewCameraMatrix + initUndistortRectifyMap + remap
    分"算表"和"查表"两步，是工程/视频里的标准做法。
    关键在 alpha 参数，它决定矫正后保留多少视野：
      alpha = 0 -> 把黑边全部裁掉，画面干净，但视野变小（等于放大截取）
      alpha = 1 -> 保留原来的全部视野，四周会留黑色边
      alpha 取 0~1 之间 -> 折中
    并且 initUndistortRectifyMap 只跟 K/D/图片尺寸有关，跟图片内容无关，
    所以 19 张图只要算一次表，之后每张用 remap 查表就行，快得多。
    本文件批量处理用的就是方法 B（alpha=0）。

====================================================================
"""
import os
import sys
import numpy as np
import cv2 as cv
import glob as gb

# ---------------- 路径配置 ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
PARAM_FILE = os.path.join(BASE, "标定参数.npz")      # lamb.py 存下来的 K、D
OUT_DIR = os.path.join(BASE, "去畸变结果")            # 批量矫正后的图片
CMP_FILE = os.path.join(BASE, "去畸变对比.jpg")       # 四栏对比图
SIMPLE_FILE = os.path.join(BASE, "去畸变_简单版.jpg")  # 方法 A 的示例产物

PREVIEW = "--no-preview" not in sys.argv   # 加这个参数就不弹窗，方便批处理


def read_image(path):
    """读取图片。Windows 上 cv.imread 遇到中文路径会返回 None，
    所以先用 np.fromfile 把文件读成字节流，再交给 imdecode 解码。"""
    return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)


def save_image(path, img, quality=95):
    """保存图片，同样是为了绕过中文路径。imencode 先压成 jpg 字节，
    再用 ndarray.tofile() 写盘。"""
    cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, quality])[1].tofile(path)


def load_params(path):
    """
    读标定参数。

    npz 是 numpy 的压缩包格式，里面按名字存了几个数组，用 d["名字"] 取。
    这里取三样东西：
      mtx  : (3,3) 内参矩阵 K
      dist : 畸变系数 D，可能是 (1,5) 也可能是 (5,)，统一拉成一维
      size : (宽, 高)，必须和要矫正的图片尺寸一致
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "找不到 %s\n请先完整运行一次 lamb.py（一路按到看完，别按空格），"
            "它会把 K、D 存成这个文件。" % path)

    d = np.load(path)
    mtx = d["mtx"]
    dist = d["dist"].reshape(-1)          # 拉平成 5 个数，避免形状差异
    size = (int(d["width"]), int(d["height"]))
    return mtx, dist, size


# ====================================================================
# 方法 A：最简单的写法，一行完事
# ====================================================================
def undistort_simple(img, mtx, dist):
    """
    cv.undistort 把【反向映射 + 插值】全封装好了。

    参数里没给 newCameraMatrix，它就拿原 K 当"矫正后的内参"：
    图片尺寸不变、视野基本不变，但四角可能出现黑色无效区。
    适合只处理一两张图、或者只想快速看效果的场合。
    """
    return cv.undistort(img, mtx, dist)


# ====================================================================
# 方法 B：算表 + 查表（批量 / 视频的标准做法）
# ====================================================================
def build_maps(mtx, dist, size, alpha):
    """
    第一步：算出映射表。注意这一步和"具体哪张图片"完全无关，
    只取决于 K、D、图片尺寸和 alpha，所以 19 张图算一次就够。

    返回：
      map1, map2 : 两张和图片一样大的查表。
                   矫正图 (u,v) 处该去原图哪个位置取色，就查这两张表。
      new_K      : 矫正后使用的新内参矩阵
      roi        : (x, y, w, h)，alpha=1 时用来把四周黑边裁掉的矩形
    """
    # getOptimalNewCameraMatrix：按 alpha 决定"矫正后的新内参"。
    #   第 4 个参数 alpha：0=尽量裁掉黑边，1=保留全部视野
    #   第 5 个参数 newImgSize：输出图尺寸，这里保持和原图一样
    # 返回的 roi 是"有效像素区域"，alpha=0 时通常覆盖整幅图。
    new_K, roi = cv.getOptimalNewCameraMatrix(mtx, dist, size, alpha, size)

    # initUndistortRectifyMap：真正把畸变公式展开成两张查表。
    #   第 3 个参数 R 传 None：这里不做双目/旋转矫正，只是单目去畸变
    #   第 6 个参数 m1type 用 CV_32FC1：表里直接存浮点坐标，最直观
    #      （想更快可以用 CV_16SC2，但表的内容就不适合拿来讲解了）
    map1, map2 = cv.initUndistortRectifyMap(
        mtx, dist, None, new_K, size, cv.CV_32FC1)
    return map1, map2, new_K, roi


def apply_maps(img, map1, map2):
    """
    第二步：用表做矫正。

    cv.remap 就是"照着表搬像素"：对输出图每个位置，查 map1/map2 得到
    原图坐标，再用双线性插值 (INTER_LINEAR) 取颜色。
    因为表已经算好了，这一步非常快 —— 视频每一帧都调它也没问题。
    """
    return cv.remap(img, map1, map2, cv.INTER_LINEAR)


def make_comparison(panels, labels, tile_w=560):
    """
    把几张图横着拼成一张对比图，每栏顶部加一条黑色标签栏。
    cv.hconcat 要求所有小图高度一致，所以这里统一按宽度等比缩放。
    """
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
    # ---------- 第 1 步：把 K、D 取回来 ----------
    mtx, dist, size = load_params(PARAM_FILE)
    print("读取标定参数完成")
    print("图像尺寸 (宽, 高):", size)
    print("内参矩阵 K:\n", mtx)
    print("畸变系数 D = [k1, k2, p1, p2, k3]:\n", dist)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 第 2 步：把映射表算好（只算一次！） ----------
    # alpha=0 用于批量输出：没有黑边，最干净，但视野会被裁小一点。
    map0_x, map0_y, new_K0, roi0 = build_maps(mtx, dist, size, alpha=0)
    # alpha=1 只用来做对比展示：保留全部视野，所以四周会有黑边。
    map1_x, map1_y, new_K1, roi1 = build_maps(mtx, dist, size, alpha=1)

    print("alpha=0 的新内参主点: cx=%.2f, cy=%.2f" % (new_K0[0, 2], new_K0[1, 2]))
    print("alpha=0 有效区域 roi (x,y,w,h):", roi0)
    print("alpha=1 有效区域 roi (x,y,w,h):", roi1,
          "  <- alpha=1 时按这个矩形裁，就能把黑边去掉")

    # ---------- 第 3 步：逐张图矫正 ----------
    images = sorted(gb.glob(os.path.join(IMG_DIR, "*.jpg")))
    print("\n待处理图片:", len(images), "张")

    example = None       # 留第一张，用来做四栏对比图
    for fname in images:
        img = read_image(fname)
        if img is None:
            print("读取失败，跳过:", os.path.basename(fname))
            continue

        # K、D 是在固定分辨率下标出来的，图片尺寸必须一致，
        # 否则映射表对不上，矫正结果会错。
        if (img.shape[1], img.shape[0]) != size:
            print("尺寸不符，跳过:", os.path.basename(fname), img.shape[1::-1])
            continue

        # 批量输出用 alpha=0：直接查表，几乎不花时间
        fixed = apply_maps(img, map0_x, map0_y)
        save_image(os.path.join(OUT_DIR, os.path.basename(fname)), fixed)

        if example is None:
            example = {
                "orig": img,                                    # ① 原图（有畸变）
                "simple": undistort_simple(img, mtx, dist),     # ② 一行版，四角可能有黑边
                "a0": fixed,                                    # ③ alpha=0，裁掉黑边
                "a1": apply_maps(img, map1_x, map1_y),          # ④ alpha=1，保留视野但有黑边
            }

    print("批量结果已存到:", OUT_DIR)

    # ---------- 第 4 步：单独存一份"方法 A"的效果 ----------
    # 它是 cv.undistort 不指定新内参的结果，和 ③ 的差别主要在
    # 视野大小与四角黑边，放在一起看最直观。
    if example is not None:
        save_image(SIMPLE_FILE, example["simple"])
        print("方法 A 示例已存:", SIMPLE_FILE)

        # ---------- 第 5 步：拼四栏对比图 ----------
        comparison = make_comparison(
            [example["orig"], example["simple"], example["a0"], example["a1"]],
            ["1 Original (distorted)", "2 cv.undistort", "3 alpha=0", "4 alpha=1"])
        save_image(CMP_FILE, comparison)
        print("四栏对比图已存:", CMP_FILE)

        # 顺带演示一下 roi 的用法：alpha=1 留了黑边，
        # 按返回的 roi 裁一刀，就得到"既保留视野又没黑边"的效果。
        x, y, w, h = roi1
        cropped = example["a1"][y:y + h, x:x + w]
        print("alpha=1 按 roi 裁剪后的尺寸:", cropped.shape[1::-1],
              "（原始尺寸:", size, "）")

        # ---------- 第 6 步：弹窗看效果 ----------
        if PREVIEW:
            shown = cv.resize(comparison, dsize=None, fx=0.75, fy=0.75)
            cv.imshow("去畸变对比   [任意键关闭]", shown)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
