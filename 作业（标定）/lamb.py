"""
相机标定：求内参 K、畸变系数 D、重投影误差，并在每张图右上角标出这些数据。

运行：python lamb.py              逐张预览（预览图和 标定结果.jpg 完全同一套标注）
                                 按任意键看下一张，按空格键(32)提前退出且不保存文件
      python lamb.py --no-preview 不弹窗，直接算出并保存结果文件

按键规则（交互模式）：
    空格  -> 立刻退出，本次不写 标定结果.jpg / 标定结果.txt
    其它键 -> 继续看下一张；9 张全部看完才保存结果文件
"""
import os
import sys
import numpy as np
import cv2 as cv
import glob as gb

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
OUT_IMG = os.path.join(BASE, "标定结果.jpg")   # 带文字的汇总图
OUT_TXT = os.path.join(BASE, "标定结果.txt")   # 纯文本结果

PREVIEW = "--no-preview" not in sys.argv

# ---------------- 标定板参数 ----------------
chessboard_size = (9, 6)   # 内角点个数 (列, 行)，不是方块数
square_size = 22.0         # 每个方格的实际边长（mm），只影响 tvecs 的尺度

# 3D 物体点：(0,0,0) (1,0,0) ... 单位是"格"，最后乘以真实边长
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 亚像素精化的终止条件：最多迭代 30 次，或位移小于 0.001 像素就停
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 屏幕上拍的照片反光会"洗白"格子，先做局部对比度增强，检出率明显提高
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def read_image(path):
    """cv.imread 读不了中文路径，用 imdecode 绕过去"""
    return cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)


def find_corners(gray):
    """找角点并做亚像素精化；找不到返回 None"""
    ret, corners = cv.findChessboardCorners(
        clahe.apply(gray), chessboard_size, None)
    if not ret:
        return None
    return cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


# ---------------- 1. 逐张找角点（先不预览，把图留着） ----------------
# 为什么要先全部找完：右上角的 K/D/RMS 是 9 张图一起标定的结果，
# 必须等 calibrateCamera 跑完才有数，不能边找边显示。
images = sorted(gb.glob(os.path.join(IMG_DIR, "*.jpg")))
print("找到图片:", len(images))

objpoints, imgpoints = [], []   # 3D 点 / 2D 像素点，必须一一对应
used_names = []                 # 真正参与标定的文件名，和上面两个列表同序
samples = []                    # (文件名, 原图, 角点)，预览要用
image_size = None

for fname in images:
    img = read_image(fname)
    if img is None:
        print("读取失败:", os.path.basename(fname))
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]            # (宽, 高)
    elif gray.shape[::-1] != image_size:
        print("注意：这张尺寸和第一张不同，跳过 ->", os.path.basename(fname))
        continue

    corners = find_corners(gray)
    if corners is None:
        print("未检测到角点:", os.path.basename(fname))
        continue

    objpoints.append(objp)
    imgpoints.append(corners)
    used_names.append(os.path.basename(fname))
    samples.append((os.path.basename(fname), img, corners))

print("有效图片:", len(objpoints))
if len(objpoints) < 3:
    raise RuntimeError("有效样本太少，标定没有意义")

# ---------------- 2. 标定 ----------------
rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, image_size, None, None)

print("内参矩阵 K:\n", mtx)
print("畸变系数 D:\n", dist.ravel())
print("重投影误差 RMS: %.4f px" % rms)

# 每张图各自的平均重投影误差，写进 txt 方便检查哪张拍得差
per_img_err = []
for i in range(len(objpoints)):
    proj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    # 两个数组都拉成 (N,2) 再算逐点距离，避免 cv.norm 对形状/类型的严格检查
    diff = proj.reshape(-1, 2) - imgpoints[i].reshape(-1, 2)
    per_img_err.append(float(np.linalg.norm(diff, axis=1).mean()))

# ---------------- 3. 结果文字（预览图和结果图共用这一份） ----------------
k1, k2, p1, p2, k3 = dist.ravel()[:5]
text_cn = [
    "标定结果",
    "内角点 %dx%d    方格边长 %g mm    样本 %d 张"
    % (chessboard_size[0], chessboard_size[1], square_size, len(objpoints)),
    "",
    "内参矩阵 K =",
    "[%9.2f %9.2f %9.2f]" % (mtx[0, 0], mtx[0, 1], mtx[0, 2]),
    "[%9.2f %9.2f %9.2f]" % (mtx[1, 0], mtx[1, 1], mtx[1, 2]),
    "[%9.2f %9.2f %9.2f]" % (mtx[2, 0], mtx[2, 1], mtx[2, 2]),
    "",
    "畸变系数 D = [k1, k2, p1, p2, k3]",
    "             = [%.6f, %.6f, %.6f, %.6f, %.6f]" % (k1, k2, p1, p2, k3),
    "",
    "重投影误差 RMS = %.4f px" % rms,
]
# 没装 Pillow 时的英文降级版（OpenCV 的 putText 画不了中文）
text_en = [
    "Calibration Result",
    "K = [%9.2f %9.2f %9.2f]" % (mtx[0, 0], mtx[0, 1], mtx[0, 2]),
    "    [%9.2f %9.2f %9.2f]" % (mtx[1, 0], mtx[1, 1], mtx[1, 2]),
    "    [%9.2f %9.2f %9.2f]" % (mtx[2, 0], mtx[2, 1], mtx[2, 2]),
    "D = [%.6f, %.6f, %.6f, %.6f, %.6f]" % (k1, k2, p1, p2, k3),
    "RMS reprojection error = %.4f px" % rms,
]


def draw_text_pil(img_bgr, lines, margin=36, line_h=44, size=32):
    """中文标注：PIL + 微软雅黑，右上角半透明黑底"""
    from PIL import Image, ImageDraw, ImageFont

    def font_for(line):
        # 含中文用微软雅黑，纯数字/英文用等宽字体，矩阵才能对齐
        if any("\u4e00" <= ch <= "\u9fff" for ch in line):
            path = r"C:\Windows\Fonts\msyh.ttc"
        else:
            path = r"C:\Windows\Fonts\consola.ttf"
        return ImageFont.truetype(path, size)

    pil = Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    fonts = [font_for(t) for t in lines]
    widths = [d.textlength(t, font=f) for t, f in zip(lines, fonts)]
    tw = max(widths) + 16       # 多留一点，防止等宽字体的右括号贴边
    th = line_h * len(lines)
    pad = 20
    w = pil.size[0]
    x1, y1 = w - margin - tw - 2 * pad, margin
    d.rounded_rectangle([x1, y1, x1 + tw + 2 * pad, y1 + th + pad],
                        radius=14, fill=(0, 0, 0, 165))

    y = y1 + pad // 2
    for i, (text, f) in enumerate(zip(lines, fonts)):
        color = (90, 235, 255, 255) if i == 0 else (255, 255, 255, 255)
        d.text((x1 + pad, y), text, font=f, fill=color)
        y += line_h

    out = Image.alpha_composite(pil, overlay).convert("RGB")
    return cv.cvtColor(np.array(out), cv.COLOR_RGB2BGR)


def draw_text_cv(img_bgr, lines):
    """英文降级标注：cv.putText + 混一层黑底"""
    result = img_bgr.copy()
    font_scale, thick, line_h = 0.85, 2, 44
    tw = max(cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, font_scale, thick)[0][0]
             for t in lines)
    x = result.shape[1] - tw - 56
    y = 80
    shade = result.copy()
    cv.rectangle(shade, (x - 24, y - 42),
                 (x + tw + 24, y + line_h * (len(lines) - 1) + 18),
                 (0, 0, 0), -1)
    result = cv.addWeighted(shade, 0.55, result, 0.45, 0)
    for i, t in enumerate(lines):
        color = (90, 235, 255) if i == 0 else (255, 255, 255)
        cv.putText(result, t, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                   font_scale, (0, 0, 0), thick + 4, cv.LINE_AA)
        cv.putText(result, t, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thick, cv.LINE_AA)
        y += line_h
    return result


def draw_text_block(img_bgr, text_cn, text_en):
    """统一的文字块入口：有 Pillow 用中文，没有就退英文"""
    try:
        return draw_text_pil(img_bgr, text_cn)
    except ImportError:
        return draw_text_cv(img_bgr, text_en)


def make_annotated(img, corners, text_cn, text_en):
    """画角点 + 右上角结果文字块。预览窗口和 标定结果.jpg 都调它，保证一模一样"""
    out = img.copy()
    cv.drawChessboardCorners(out, chessboard_size, corners, True)
    return draw_text_block(out, text_cn, text_en)


# ---------------- 4. 交互预览：waitKey(0)，空格退出且不保存 ----------------
save_ok = True     # 只有完整看完所有图，才允许写结果文件

if PREVIEW:
    for name, img, corners in samples:
        shown = make_annotated(img, corners, text_cn, text_en)
        # 只是缩放显示尺寸，标注内容（数据）和 标定结果.jpg 完全一致
        shown = cv.resize(shown, dsize=None, fx=0.5, fy=0.5)
        cv.imshow("preview  [SPACE = quit without saving, other key = next]", shown)

        key = cv.waitKey(0) & 0xFF   # 0 = 一直等到按键为止
        if key == 32:                # 32 就是空格键
            save_ok = False
            print("按空格退出，本次不保存结果文件")
            break

    cv.destroyAllWindows()

# ---------------- 5. 保存（按空格退出时整段跳过） ----------------
if save_ok:
    # 把标定得到的 K、D 单独存一份，main.py 做去畸变时直接读，不用再标定一遍
    PARAM = os.path.join(BASE, "标定参数.npz")
    np.savez(PARAM, mtx=mtx, dist=dist,
             width=image_size[0], height=image_size[1])
    print("已保存:", PARAM)

    # 用第一张图做汇总图，和预览看到的是同一套绘制
    result_img = make_annotated(samples[0][1], samples[0][2], text_cn, text_en)
    cv.imencode(".jpg", result_img, [cv.IMWRITE_JPEG_QUALITY, 95])[1].tofile(OUT_IMG)
    print("已保存:", OUT_IMG)

    with open(OUT_TXT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(text_cn) + "\n\n每张图的重投影误差(px):\n")
        for fname, err in zip(used_names, per_img_err):
            fp.write("  %s  %.4f\n" % (fname, err))
    print("已保存:", OUT_TXT)
