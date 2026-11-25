# -*- coding: utf-8 -*-
"""
【选做】平面靶面位姿估计 —— 摄像头实时版

程序做什么
----------
摄像头每拍到一帧，程序做四件事：
    1. 在画面里找到方形靶面的四个角点（按 A→B→C→D 的固定顺序）
    2. 用 solvePnP 解出这一帧的位姿：rvec（旋转）+ tvec（平移）
    3. 把靶面自己的坐标系（X 红 / Y 绿 / Z 蓝）画到画面上
    4. 把 rvec、tvec、重投影误差贴在画面左上角

按 q 退出。

运行方式
--------
    python main.py                # 默认：纯色方块靶面
    python main.py --aruco        # 换成 ArUco 标签（更稳，顺序由库保证）
    python main.py --record       # 边跑边录 位姿录屏.mp4，退出时自动存好
    python main.py --make-target  # 生成可打印的 A4 靶面图，不需要摄像头
    python main.py --edge 80      # 靶面边长不是 100mm 时，用这个改

为什么需要相机内参
------------------
solvePnP 解的是这个方程：
    像素坐标 = 内参 K · ( R · 靶面坐标 + t )
K 负责把「相机坐标系里的三维点」压成「像素」，R、t 描述靶面相对相机的位置。
所以 K 不知道，tvec 就没法换算成毫米。本项目默认按视野角估一个 K，
够画轴、够看数值；如果你对这台摄像头做过标定，用 --calib 指定 npz 更准。
"""

import argparse
import os
import time

import cv2 as cv
import numpy as np


# ============================ 可调参数 ============================

# 靶面边长，单位毫米。改这里就等于告诉程序「我这张纸上的方块有多大」。
# 它只影响 tvec 的数值大小（300mm 还是 400mm），不影响旋转 rvec。
SQUARE_MM = 100.0

# 摄像头编号。本机只有 0 这一路能打开，插了外接摄像头再试 1。
CAM_INDEX = 0

# 想要的分辨率。摄像头只会给最接近的档位，本机最高 1280x720。
REQUEST_SIZE = (1280, 720)

# 没有标定文件时，按这个水平视野角估算内参。笔记本摄像头一般 55°~70°。
ASSUMED_HFOV = 60.0

# ArUco 使用的字典与编号（要和打印出来的标签一致）
ARUCO_DICT_ID = cv.aruco.DICT_4X4_50 if hasattr(cv, "aruco") else None
ARUCO_TARGET_ID = 0

# 重投影误差上限（像素）。解出来的位姿如果连自己那 4 个点都对不回去，
# 说明检到的根本不是靶面（显示器边框、窗框、纸边……），直接当没找到丢弃。
MAX_REPROJ_PX = 5.0

# 上一份作业（相机标定）存下来的参数，尺寸能对上就直接用
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "作业（标定）", "标定参数.npz")

# 画坐标轴的长度，取边长的 0.6 倍，太长会捅出画面
AXIS_LEN_MM = SQUARE_MM * 0.6


def open_camera(index):
    """
    打开摄像头。

    Windows 上 OpenCV 默认走 MSMF 后端，在这台机器上实测 VideoCapture(0) 要卡
    将近 53 秒才返回（读第一帧又要 5 秒），看起来像是程序死了，其实是后端在磨蹭。
    显式指定 DirectShow 是毫秒级，所以优先用它，失败再退回默认后端。
    这个坑跟算法无关，纯粹是采集层的事，但不知道的话会以为代码有 bug。
    """
    for backend, name in ((cv.CAP_DSHOW, "DirectShow"), (cv.CAP_ANY, "默认")):
        cap = cv.VideoCapture(index, backend)
        if cap.isOpened():
            print("摄像头 %d 已打开（%s 后端）" % (index, name))
            return cap
        cap.release()
    raise RuntimeError("打不开摄像头 %d：检查是否被其他程序占用，或换一个编号试"
                       % index)


# ============================ 相机内参 ============================

def camera_matrix_from_fov(width, height, hfov_deg):
    """
    没做标定时，用「水平视野角」反推一个近似内参。

    针孔模型里，画面最右边缘那条光线和光轴夹角是 hfov/2，于是
        tan(hfov/2) = (width/2) / f   →   f = (width/2) / tan(hfov/2)

    再假设像素是正方形（fx = fy）、主点在画面正中，K 就齐了。
    这个 K 有偏差时，tvec 的毫米数会差百分之几，但三轴方向是对的，
    拿来演示位姿完全够用。
    """
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)
    k = np.array([[f, 0.0, width / 2.0],
                  [0.0, f, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return k, np.zeros((5, 1), np.float64), "按 %.0f° 水平视野估算" % hfov_deg


def load_camera_matrix(width, height, calib_path=None):
    """
    有标定文件就优先用标定结果，否则退回估算。

    注意：标定参数是跟「分辨率」绑定的。同一个镜头换分辨率，焦距 fx 会按比例变、
    主点 cx/cy 也会跟着变，所以尺寸对不上时不能硬用（硬用会让位姿整体偏移）。
    这种情况直接放弃标定文件、退回估算，并把原因打出来。
    """
    path = calib_path or CALIB_PATH
    if os.path.exists(path):
        try:
            d = np.load(path)
            cw, ch = int(d["width"]), int(d["height"])
            if (cw, ch) == (width, height):
                return (d["mtx"].astype(np.float64),
                        d["dist"].reshape(-1).astype(np.float64),
                        "来自标定文件 %s" % os.path.basename(path))
            print("[提示] 标定文件是 %dx%d，摄像头是 %dx%d，尺寸不一致，"
                  "不采用（原因见 load_camera_matrix 注释）" % (cw, ch, width, height))
        except Exception as e:
            print("[提示] 读取标定文件失败：%s" % e)

    return camera_matrix_from_fov(width, height, ASSUMED_HFOV)


# ============================ 靶面的三维坐标 ============================

def target_object_points(edge_mm):
    """
    靶面四个角点在自己坐标系里的三维坐标，单位毫米。

    约定：靶面中心是原点，X 向右、Y 向上、Z 垂直纸面向外（朝相机）。
    顺序固定是「左上 → 右上 → 右下 → 左下」，对应画面里的 A→B→C→D。
    注意左上角的 Y 是 +h（因为 Y 轴朝上），这是最容易写反的一个符号。

    【为什么必须用这个顺序】
    solvePnP 带 SOLVEPNP_IPPE_SQUARE 标志时，OpenCV 规定三维点必须恰好写成
    [-h,+h,0] [+h,+h,0] [+h,-h,0] [-h,-h,0]，差一点都不行。
    顺序给错了它不会报错，而是闷头算出一个无意义的位姿：重投影误差上万像素、
    tvec 的 z 变成负数（等于说靶面在相机背后）。这种错误从现象上很难反推，
    所以规矩直接写在注释里。

    像素点也要按同样的绕向给（见 order_quad）：三维点和像素点的「转圈方向」
    必须一致，否则位姿会镜像翻转，三轴倒个个儿 —— 就是老师说的三轴翻转。
    """
    h = edge_mm / 2.0
    return np.array([[-h, h, 0.0],
                     [h, h, 0.0],
                     [h, -h, 0.0],
                     [-h, -h, 0.0]], dtype=np.float64)


# ============================ 角点排序 ============================

def order_quad(points, prev=None):
    """
    把任意给进来的 4 个点整理成固定顺序，这是整个程序最容易出错的地方。

    分两步，缺一不可：

    【第一步：统一绕向】
    用鞋带公式算有向面积 area2 = Σ(x_i·y_{i+1} - x_{i+1}·y_i)。
    它的正负号只跟「四个点转圈的方向」有关，跟从哪个点开始无关。
    这里固定取正号：像素坐标 y 朝下时，正号正好对应
    A(左上)→B(右上)→C(右下)→D(左下) 这个视觉上的顺时针方向，
    与 target_object_points 里四个三维点的绕向一致，也与 ArUco 库自带的
    角点顺序一致（这一条可以实测验证：把 ArUco 返回的角点直接算鞋带和，
    结果就是正的）。
    不统一会怎样：同一张纸换个角度，轮廓绕向可能反了，于是第 1 个点配到了
    靶面的对角点上，位姿瞬间镜像翻转，屏幕上的 X/Y 轴像照镜子一样掉个个儿——
    这就是老师说的「三轴翻转」。

    【第二步：统一起点】
    绕向对了，但「谁是第 0 个」还有 4 种可能。正方形转 90° 之后样子完全一样，
    只看一帧真的分不出来，所以：
      · 有上一帧：挑和上一帧最贴合的那个起点，保证视频里不跳
      · 第一帧  ：挑最靠左上的点当 A，也就是 min(x+y)
    这就是「位姿乱跳」的解法 —— 用时间上的连续性，代替几何上的唯一性。
    """
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)

    # 先按极角排一圈，得到一个连续的环（起点随机）
    center = p.mean(axis=0)
    angles = np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])
    p = p[np.argsort(angles)]

    # 鞋带公式：统一绕向
    area2 = 0.0
    for i in range(4):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % 4]
        area2 += x1 * y2 - x2 * y1
    if area2 < 0:
        p = p[::-1].copy()

    # 起点：贴着上一帧，或者取最左上
    if prev is None:
        p = np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)
    else:
        prev = np.asarray(prev, dtype=np.float32).reshape(4, 2)
        best_cost, best_p = None, p
        for k in range(4):
            cand = np.roll(p, -k, axis=0)
            cost = float(((cand - prev) ** 2).sum())
            if best_cost is None or cost < best_cost:
                best_cost, best_p = cost, cand
        p = best_p

    return p


# ============================ 找靶面 ============================

def find_square(gray, min_area):
    """
    在灰度图里找「一个近似的实心四边形」，找不到返回 None。

    流程：模糊 → 反相二值化 → 找轮廓 → 逼近成 4 个顶点 → 层层过滤

    为什么要反相（THRESH_BINARY_INV）：靶面是黑纸白底，反相之后靶面变成白块，
    findContours 找白色区域，轮廓正好是靶面外沿。
    阈值用 Otsu 自动定，比写死 127 抗光照变化。
    """
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # 闭运算补掉纸面的小洞、笔迹断口，免得轮廓被拆成几段
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_h, img_w = gray.shape[:2]
    max_area = img_h * img_w * 0.6      # 手持靶面不可能占满整个画面

    best, best_area = None, 0.0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:              # 太小的（噪点、键帽、鼠标）跳过
            continue
        if area > max_area:              # 太大的也不是靶面（显示器、墙面、窗户）
            continue

        # 贴边的轮廓一律丢掉。画面最外圈那条"边框"本身就能被逼近成 4 个点，
        # 摄像头对着空房间时会稳定地误检成一个大四边形，位姿完全是无意义的。
        bx, by, bw, bh = cv.boundingRect(cnt)
        if bx <= 2 or by <= 2 or bx + bw >= img_w - 2 or by + bh >= img_h - 2:
            continue

        # 用周长的一定比例做逼近精度：四边形压成 4 个顶点，圆或不规则块会留下
        # 5 个以上顶点，于是被下面的 len != 4 筛掉
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue

        # 凸性只能判在「逼近后的 4 个点」上。
        # 直接判原始轮廓会踩坑：像素是方格拼出来的，斜边天然带锯齿，
        # 几百个点的原始轮廓几乎永远被判成「非凸」，好好的靶面会被全部过滤掉。
        if not cv.isContourConvex(approx):
            continue

        # 面积 / 外接矩形面积，接近 1 才像个实心块（凹形、细长条都会被压下去）
        (_, _), (rw, rh), _ = cv.minAreaRect(cnt)
        if rw * rh <= 0 or area / (rw * rh) < 0.72:
            continue

        if area > best_area:             # 画面里可能有多个四边形，取最大的
            best, best_area = approx.reshape(4, 2).astype(np.float32), area

    return best


def refine_corners(gray, pts):
    """
    亚像素精化：把整数像素角点再往里挤到 0.1 像素级。

    approxPolyDP 给的是轮廓上的整数坐标，天生带 ±1 像素抖动；而只拿 4 个点
    去解 6 个自由度（3 旋转 + 3 平移），误差会被放大，视频里就是坐标轴抖。
    精化之后明显稳。失败就原样返回，不影响主流程。
    """
    try:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        refined = cv.cornerSubPix(gray, pts.reshape(-1, 1, 2).copy(),
                                  (5, 5), (-1, -1), criteria)
        return refined.reshape(4, 2).astype(np.float32)
    except cv.error:
        return pts


def find_marker(gray):
    """
    ArUco 模式：直接返回标签角点（顺序由库保证 左上→右上→右下→左下）。

    OpenCV 4.7 之后旧的 estimatePoseSingleMarkers 就被标废弃、5.x 里删掉了，
    现在统一用 ArucoDetector.detectMarkers 拿角点，再自己调 solvePnP，
    这样方块靶面和 ArUco 靶面走同一套解位姿代码，改动最小。
    """
    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(ARUCO_DICT_ID),
        cv.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None

    for i, marker_id in enumerate(ids.ravel()):
        if int(marker_id) == ARUCO_TARGET_ID:
            return corners[i].reshape(4, 2).astype(np.float32), int(marker_id)
    return None, None


# ============================ 解位姿 ============================

def solve_pose(objp, imgp, k, dist):
    """
    调用 solvePnP 解出这一帧的旋转和平移。

    输入  objp：靶面 4 个角点的三维坐标（毫米）
          imgp：这 4 个角点在画面里的像素坐标，顺序必须与 objp 严格对应
          k   ：相机内参
    输出  rvec：旋转向量（3 个数；方向是旋转轴，长度是转角弧度）
          tvec：平移向量（毫米），即「靶面中心在相机坐标系里的位置」
          err ：把解出的位姿重新投影回画面，与实际角点的平均偏差（像素）

    这里用 SOLVEPNP_IPPE_SQUARE：专门为「正方形四个角点」设计的解法。
    平板靶面本来就有两个数学上都说得通的位姿（一正一反），通用迭代法可能
    收敛到反的那个，IPPE 会稳定地挑相机前面那个。
    """
    ok, rvec, tvec = cv.solvePnP(objp, imgp, k, dist,
                                 flags=cv.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None, float("inf")

    # 重投影误差：拿解出的位姿把三维点再投一次，看和检测到的角点差多少像素。
    # 这个数最能说明「解得好不好」，一般 < 1.5 像素就没问题。
    projected, _ = cv.projectPoints(objp, rvec, tvec, k, dist)
    err = float(np.linalg.norm(projected.reshape(-1, 2) - imgp.reshape(-1, 2),
                               axis=1).mean())
    return rvec, tvec, err


# ============================ 画面绘制 ============================

def draw_panel(frame, lines, org=(16, 16), scale=0.55, pad=8):
    """
    在左上角贴一块半透明黑底白字，保证任何背景上都看得清。

    cv.putText 只能写英文数字（画不了中文），所以屏幕上的字都是英文缩写。
    不影响交作业，录屏里 rvec / tvec 的数值才是重点。
    """
    if not lines:
        return frame

    font, thickness = cv.FONT_HERSHEY_SIMPLEX, 1
    sizes = [cv.getTextSize(t, font, scale, thickness)[0] for t in lines]
    line_h = max(s[1] for s in sizes) + 10      # 行高：字高 + 行间距
    w = max(s[0] for s in sizes) + pad * 2
    h = line_h * len(lines) + pad * 2
    x, y = org

    # 半透明黑底：只处理这一小块 ROI，比全图 addWeighted 快得多
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    roi = frame[y:y2, x:x2]
    if roi.size:
        black = np.zeros_like(roi)
        cv.addWeighted(black, 0.55, roi, 0.45, 0, roi)

    # putText 的 y 是「文字基线」，不是文字顶部，所以基线下移 line_h 再回退 4px，
    # 让每一行都落在自己那一格里，既不会顶到上一行，也不会在底部留大片空白
    for i, (text, size) in enumerate(zip(lines, sizes)):
        base_y = y + pad + line_h * (i + 1) - 4
        cv.putText(frame, text, (x + pad, base_y), font, scale, (0, 0, 0),
                   thickness + 2, cv.LINE_AA)       # 先描黑边
        cv.putText(frame, text, (x + pad, base_y), font, scale, (255, 255, 255),
                   thickness, cv.LINE_AA)           # 再写白字
    return frame


def draw_target(frame, corners, rvec, tvec, k, dist, ok):
    """
    画三件东西：靶面轮廓 + 四个角点编号 A/B/C/D + 三维坐标轴。
    """
    pts = np.rint(corners.reshape(-1, 2)).astype(np.int32)

    # 1) 四边连起来，方便肉眼确认「程序认的是不是这张纸」
    cv.polylines(frame, [pts.reshape(-1, 1, 2)], True, (0, 220, 255), 2, cv.LINE_AA)

    # 2) 角点编号，颜色一一对应，录屏里能看出顺序有没有翻
    colors = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255)]
    for i, (px, py) in enumerate(pts):
        cv.circle(frame, (int(px), int(py)), 5, colors[i], -1, cv.LINE_AA)
        label = chr(ord("A") + i)
        cv.putText(frame, label, (px + 8, py - 8), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(frame, label, (px + 8, py - 8), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, colors[i], 2, cv.LINE_AA)

    # 3) 坐标轴：drawFrameAxes 内部就是拿 rvec/tvec 把三条轴线投影下来，
    #    颜色固定 X 红、Y 绿、Z 蓝。靶面正对镜头时 Z 几乎指着眼睛，会缩成一个点，
    #    这是正常的，把纸稍微倾斜就能看到它伸出来。
    if ok and rvec is not None:
        cv.drawFrameAxes(frame, k, dist, rvec, tvec, AXIS_LEN_MM, 3)

    return frame


# ============================ 生成可打印靶面 ============================

def save_png(path, img):
    """
    存图。这里不用 cv.imwrite 是因为它只支持英文路径：
    遇到「作业（录屏识别）」这种含中文的目录，imwrite 会悄悄返回 False、
    什么文件都不生成，而且不抛异常 —— 上一份作业读图时踩的也是同一个坑，
    只是方向相反。imencode 先在内存里编好 PNG，再用 numpy 的 tofile 落盘，
    中文路径就没问题了。
    """
    ok, buf = cv.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 编码失败: %s" % path)
    buf.tofile(path)


def make_targets(out_dir):
    """
    生成两张 A4、300dpi 的可打印靶面图：
        target_square.png   纯色方块（默认模式用）
        target_aruco.png    ArUco 标签（--aruco 模式用）

    300dpi 下 A4 是 2480×3508 像素，1 毫米 = 11.81 像素，100mm 方块就是 1181 像素。
    打印时必须选「实际大小 / 100%」，不要选「适应页面」——后者会把方块缩成 96% 左右，
    tvec 的毫米数就全偏了。图里画了一把毫米刻度尺，打印完拿尺子量一下便知有没有缩。
    """
    dpi = 300.0
    mm2px = dpi / 25.4
    page_w, page_h = int(210 * mm2px), int(297 * mm2px)
    side = int(round(SQUARE_MM * mm2px))

    # ---------- 纯色方块 ----------
    canvas = np.full((page_h, page_w), 255, np.uint8)
    cx, cy = page_w // 2, page_h // 2 - int(20 * mm2px)
    x0, y0 = cx - side // 2, cy - side // 2
    cv.rectangle(canvas, (x0, y0), (x0 + side, y0 + side), 0, -1)

    cv.putText(canvas, "print at 100%", (x0, y0 - int(12 * mm2px)),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv.LINE_AA)

    # 毫米刻度尺：打印缩放没设对，拿尺子一量就知道差多少
    ruler_y = y0 + side + int(15 * mm2px)
    cv.line(canvas, (x0, ruler_y), (x0 + side, ruler_y), 0, 3)
    for i in range(0, int(SQUARE_MM) + 1, 10):
        px = x0 + int(round(i * mm2px))
        cv.line(canvas, (px, ruler_y - 8), (px, ruler_y + 8), 0, 2)
    cv.putText(canvas, "%d mm" % int(SQUARE_MM),
               (x0, ruler_y + int(12 * mm2px)), cv.FONT_HERSHEY_SIMPLEX,
               1.2, 0, 3, cv.LINE_AA)

    p1 = os.path.join(out_dir, "target_square.png")
    save_png(p1, canvas)
    print("已生成:", p1)

    # ---------- ArUco 标签 ----------
    if ARUCO_DICT_ID is not None:
        dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        # generateImageMarker 出来的图「整张就是标签本身」（含最外圈黑框），
        # 所以尺寸必须正好等于边长；之前多留边会让黑框小于 SQUARE_MM，
        # tvec 的毫米数就会整体偏小，属于隐蔽的尺度错误。
        marker = cv.aruco.generateImageMarker(dictionary, ARUCO_TARGET_ID, side)
        canvas2 = np.full((page_h, page_w), 255, np.uint8)
        my, mx = y0, cx - marker.shape[1] // 2
        canvas2[my:my + marker.shape[0], mx:mx + marker.shape[1]] = marker
        cv.putText(canvas2, "ArUco 4x4_50  id=%d  side %d mm"
                   % (ARUCO_TARGET_ID, int(SQUARE_MM)),
                   (x0, my + marker.shape[0] + int(14 * mm2px)),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv.LINE_AA)
        p2 = os.path.join(out_dir, "target_aruco.png")
        save_png(p2, canvas2)
        print("已生成:", p2)


# ============================ 主流程 ============================

def parse_args():
    p = argparse.ArgumentParser(description="平面靶面位姿估计（PnP）")
    p.add_argument("--aruco", action="store_true", help="用 ArUco 标签代替纯色方块")
    p.add_argument("--record", action="store_true", help="同时录制 位姿录屏.mp4")
    p.add_argument("--make-target", action="store_true", help="只生成可打印靶面图后退出")
    p.add_argument("--calib", default=None, help="指定相机标定 npz 文件")
    p.add_argument("--edge", type=float, default=None, help="靶面边长（毫米）")
    return p.parse_args()


def main():
    args = parse_args()

    # drawFrameAxes 在轴线端点跑出画面时会刷一堆 WARN，一分钟能刷几百行，
    # 把有用的信息全冲掉了。这里把 OpenCV 日志压到只报错。
    try:
        cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass

    # 命令行给了 --edge 就覆盖全局边长；打印靶面图那条分支也要用这个值，
    # 所以放在 make_target 之前。
    global SQUARE_MM, AXIS_LEN_MM
    if args.edge is not None:
        SQUARE_MM = args.edge
    edge_mm = SQUARE_MM
    AXIS_LEN_MM = edge_mm * 0.6

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.make_target:
        make_targets(base_dir)
        return

    # 两套靶面的角点顺序一致（都是 左上→右上→右下→左下），所以共用一套三维点
    objp = target_object_points(edge_mm)

    cap = open_camera(CAM_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, REQUEST_SIZE[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, REQUEST_SIZE[1])

    writer = None
    prev_corners = None       # 上一帧的角点顺序，用来保持位姿连续
    cur_size = None
    k = dist = None
    fps_t0, fps_cnt, fps = None, 0, 0.0

    print("摄像头已打开，按 q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读不到画面，退出")
            break
        h, w = frame.shape[:2]

        # 内参只跟分辨率有关，分辨率没变就不重复算
        if (w, h) != cur_size:
            k, dist, k_src = load_camera_matrix(w, h, args.calib)
            cur_size = (w, h)
            print("画面尺寸 %dx%d，内参来源：%s" % (w, h, k_src))
            print("  fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f"
                  % (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
            if args.record:
                writer = cv.VideoWriter(os.path.join(base_dir, "位姿录屏.mp4"),
                                        cv.VideoWriter_fourcc(*"mp4v"),
                                        25.0, (w, h))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ---- 找角点 ----
        if args.aruco:
            found, _ = find_marker(gray)
        else:
            found = find_square(gray, min_area=w * h * 0.004)

        rvec = tvec = None
        err = float("nan")

        if found is not None:
            if not args.aruco:
                found = refine_corners(gray, found)
            corners = order_quad(found, prev_corners)
            rvec, tvec, err = solve_pose(objp, corners, k, dist)

            # 误差闸门：位姿连自己那 4 个角点都投不回去，就说明这四边根本不是靶面
            # （显示器边框、窗框、纸边都很容易被当成四边形）。
            # 不过闸门只在画轴之前判，避免屏幕上出现离谱的坐标轴。
            if rvec is None or err > MAX_REPROJ_PX:
                prev_corners = None
                rvec = tvec = None
                err = float("nan")
            else:
                prev_corners = corners
                draw_target(frame, corners, rvec, tvec, k, dist, True)
        else:
            prev_corners = None      # 丢了靶面就忘了上一帧，重新出现时重新定位起点

        # ---- 左上角数据面板 ----
        lines = ["target %s  L=%.0fmm   fps %.0f"
                 % ("aruco4x4_50" if args.aruco else "square", edge_mm, fps)]
        if rvec is not None:
            r, t = rvec.ravel(), tvec.ravel()
            lines += ["rvec (rad): %.3f  %.3f  %.3f" % (r[0], r[1], r[2]),
                      "tvec (mm):  %.1f  %.1f  %.1f" % (t[0], t[1], t[2]),
                      "distance: %.0f mm" % np.linalg.norm(t),
                      "reproj err: %.2f px" % err]
        else:
            lines += ["target not found",
                      "hold the printed square in front of camera"]
        draw_panel(frame, lines)

        # ---- 帧率统计，指数平滑一下免得数字乱跳 ----
        if fps_t0 is None:
            fps_t0 = time.time()
        fps_cnt += 1
        if fps_cnt >= 10:
            dt = time.time() - fps_t0
            now = fps_cnt / dt if dt > 0 else 0.0
            fps = now if fps == 0.0 else fps * 0.7 + now * 0.3
            fps_t0, fps_cnt = time.time(), 0

        if writer is not None:
            writer.write(frame)

        cv.imshow("pose estimation   [q to quit]", frame)
        if (cv.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print("录屏已保存:", os.path.join(base_dir, "位姿录屏.mp4"))
    cv.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
