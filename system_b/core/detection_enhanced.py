# ====================================================================================
# 模块名称: detection_enhanced.py
# 功能描述: ESP32-CAM 农作物病虫害检测系统的核心算法模块
#
# 本模块实现了基于 OpenCV 的完整病虫害检测流水线：
#   1. 图像获取（从 ESP32-CAM 摄像头或本地测试图片）
#   2. 叶片分割（基于绿色通道优势 + 饱和度筛选的 AND 策略）
#   3. 病斑检测（基于 HSV 颜色空间中棕色/黄色区域识别）
#   4. 虫害检测（基于白色低饱和度点识别）
#   5. 严重程度分级（基于数量和面积占比的双阈值体系）
#   6. 结果可视化标注（在原始图像上绘制轮廓、边框和文字信息）
#
# 核心设计思想：
#   - 分辨率自适应：参数根据图像分辨率动态调整，兼容 QVGA 到 UXGA 等多种分辨率
#   - AND 策略：叶片分割采用纯 AND 逻辑（绿色优势 ∩ 饱和度下限），避免 OR 合并引入误判
#   - 形态学补偿：通过闭运算填充叶片高光空洞，而非引入额外判断层
# ====================================================================================

import cv2
import numpy as np
import requests
import os


# ====================================================================================
# 全局常量配置
# ====================================================================================

# ESP32-CAM 的图像捕获接口地址
# ESP32-CAM 固件（如 ArduinoWebcamServer）启动后会在局域网内提供 HTTP 图像流，
# 通过访问该 URL 可获取一帧 JPEG 图片用于后续病虫害检测分析。
# 默认地址需根据实际网络环境修改为 ESP32-CAM 的局域网 IP。
URL = "http://【请修改：摄像头IP地址】/capture"

# 本地测试图片路径
# 当设置为有效文件路径时，系统将跳过摄像头获取，直接使用本地图片进行检测调试。
# 设为 None 表示正常模式——从 ESP32-CAM 摄像头实时获取图像。
# 典型用法：调试时设为本地测试图片路径即可。
TEST_IMAGE_PATH = None


# ====================================================================================
# 检测参数配置类
# ====================================================================================

class DetectionConfig:
    """
    简化版检测参数配置 - 方向A：单一可靠方法 + 分辨率自适应

    【叶片分割策略 - AND 策略说明】
    所谓 "AND 策略" 是指叶片像素的判定需要同时满足两个独立条件：
      条件1: 绿色通道优势（G - max(R, B) > GREEN_ADV）——像素在绿色通道上明显强于红/蓝
      条件2: 饱和度下限（S >= SAT_MIN）——像素具有足够的色彩饱和度
    只有 条件1 AND 条件2 同时为真，才认定为叶片像素。
    与之相对的 "OR 策略" 会将两个条件的结果取并集，虽然能检测更多区域，
    但也容易将背景中的绿色物体（如绿色桌面、键盘背光）误判为叶片。
    纯 AND 逻辑保证了高精确率（precision），宁可漏检也不误判。

    【高光补偿策略说明】
    叶片表面常有反光/高光区域，这些区域的绿色特征会被白色光冲淡，
    导致在步骤2中被排除。本模块不引入额外的"高光检测层"，
    而是通过形态学闭运算（MORPH_CLOSE）直接从周围绿色区域"填充"空洞，
    简单高效，避免增加判断复杂度。

    【分辨率自适应说明】
    resolution_adaptive 开关控制参数是否随图像分辨率动态调整。
    ESP32-CAM 支持多种分辨率（QVGA 320×240 到 UXGA 1600×1200），
    固定阈值在不同分辨率下表现差异巨大：
      - 面积比例：叶片在 UXGA 中像素数是 QVGA 的 25 倍，固定比例阈值会过于严格
      - 形态学核：3×3 的核在 QVGA 中合适，但在 UXGA 中太小无法有效去噪
    开启自适应后，系统会根据当前分辨率自动计算合适的参数值。
    """

    def __init__(self, resolution_adaptive=True):
        """
        初始化检测参数配置。

        :param resolution_adaptive: 是否启用分辨率自适应（默认 True）。
                                    设为 True 时，形态学核大小、面积阈值等参数会根据图像分辨率
                                    动态调整；设为 False 时使用固定基准值，适合分辨率固定的场景。
        """
        # === 叶片分割核心参数（AND策略） ===
        # 这两个参数共同构成叶片像素的判定条件：
        #   绿色优势 > GREEN_ADV  AND  饱和度 >= SAT_MIN  →  认定为叶片像素
        self.GREEN_ADV = 12          # 绿色通道优势阈值：G 通道值减去 max(R, B) 的差值必须大于 12
                                     # 该值越大越严格（只识别非常绿的区域），越小越宽松（浅绿也会被识别）
                                     # 12 是经过实验调优的平衡值，能区分健康叶片和大多数背景
        self.SAT_MIN = 25            # 饱和度下限（HSV 中 S 通道，范围 0~255）：
                                     # 饱和度低于 25 的像素接近灰度（如键盘、水泥地面、阴影区域），
                                     # 即使它们的绿色通道偏强也不应被识别为叶片。
                                     # 25 能有效过滤低饱和度背景，同时不影响正常叶片的检测。

        # === 形态学参数（基础值，自适应计算后覆盖） ===
        # 以下为基础默认值，当 resolution_adaptive=True 时，
        # get_adaptive_params() 会根据分辨率返回更合适的值覆盖这些基础值。
        self.MORPH_KERNEL_SIZE = 3   # 基础形态学核大小（3×3），用于开运算去噪和膨胀连接
        self.DILATE_ITERATIONS = 1   # 膨胀迭代次数，1 次适度膨胀可连接相邻叶片碎片而不过度扩张

        # === 病斑检测参数（HSV 颜色空间范围，不随分辨率变化） ===
        # 病斑在 HSV 空间中表现为棕色和黄色两类区域：
        # - 棕色病斑（早期/坏死）：色相 H 偏低（接近红色端），饱和度适中
        # - 黄色病斑（黄化/褪绿）：色相 H 在黄光区间，饱和度适中
        # 注意：OpenCV 的 H 通道范围是 0~180（不是 0~360），S/V 范围是 0~255
        self.BROWN_H_MIN = 0         # 棕色色相下限（H=0 对应红色端）
        self.BROWN_H_MAX = 15        # 棕色色相上限（H=15 约对应橙红色）
        self.BROWN_S_MIN = 50        # 棕色饱和度下限：过滤低饱和度的灰色/白色干扰
        self.YELLOW_H_MIN = 18       # 黄色色相下限（H=18 约对应橙黄色）
        self.YELLOW_H_MAX = 32       # 黄色色相上限（H=32 约对应黄绿色）
        self.YELLOW_S_MIN = 50       # 黄色饱和度下限：与棕色相同，过滤低饱和度干扰

        # === 虫害检测参数（HSV 颜色空间范围，不随分辨率变化） ===
        # 虫害（如白粉虱、蚜虫蜕皮等）在图像中表现为叶片上的白色小点：
        # - 白色在 HSV 中特征为：饱和度极低（接近 0）且亮度极高（接近 255）
        # - 通过极严格的 S 上限和 V 下限，只识别近乎纯白的亮点
        self.PEST_S_MAX = 15         # 虫害饱和度上限：S <= 15 表示几乎无色（纯白/灰白），
                                     # 该值极低以确保只有真正的白色才被识别，避免误判浅色叶片
        self.PEST_V_MIN = 220        # 虫害亮度下限：V >= 220 表示非常亮（接近纯白），
                                     # 配合 S_MAX 形成 "高亮且无色" 的白色检测条件

        # === 分级阈值参数（双阈值体系：数量 OR 占比） ===
        # 分级系统对病斑和虫害分别设定 "数量阈值" 和 "面积占比阈值"，
        # 任一指标超过阈值即触发对应等级（取最高等级）。
        # 例如：即使病斑数量只有 2 个，但如果面积占比超过 5%，也会触发"注意"等级。
        # 这种双阈值设计能同时覆盖 "少而大" 和 "多而小" 两种病虫害模式。

        # -- 病斑分级阈值 --
        self.NOTICE_DISEASE_COUNT = 3      # 注意级：病斑数量 >= 3
        self.NOTICE_DISEASE_RATIO = 0.05   # 注意级：病斑面积占全图比例 > 5%
        self.WARNING_DISEASE_COUNT = 5     # 警告级：病斑数量 >= 5
        self.WARNING_DISEASE_RATIO = 0.08  # 警告级：病斑面积占全图比例 > 8%
        self.SERIOUS_DISEASE_COUNT = 8     # 严重级：病斑数量 >= 8
        self.SERIOUS_DISEASE_RATIO = 0.15  # 严重级：病斑面积占全图比例 > 15%

        # -- 虫害分级阈值 --
        self.NOTICE_PEST_COUNT = 5         # 注意级：虫害白点数量 >= 5（虫害无占比阈值，因为白点通常很小）
        self.WARNING_PEST_COUNT = 8        # 警告级：虫害白点数量 >= 8
        self.WARNING_PEST_RATIO = 0.02     # 警告级：虫害面积占全图比例 > 2%
        self.SERIOUS_PEST_COUNT = 15       # 严重级：虫害白点数量 >= 15
        self.SERIOUS_PEST_RATIO = 0.05     # 严重级：虫害面积占全图比例 > 5%

        # -- 绿色占比阈值（反映叶片整体健康程度） --
        self.GREEN_RATIO_NOTICE = 0.20     # 注意级：绿色占比 < 20% 表示叶片覆盖不足或大面积枯萎
        self.GREEN_RATIO_WARNING = 0.10    # 警告级：绿色占比 < 10% 表示叶片严重枯萎或几乎无健康叶片

        # === 分辨率自适应开关 ===
        self.resolution_adaptive = resolution_adaptive  # 控制是否启用分辨率自适应参数计算
        self.base_resolution = (320, 240)  # 基准分辨率：QVGA（320×240 = 76800 像素）
                                           # 所有自适应参数的基准值都是在此分辨率下调优得到的，
                                           # 其他分辨率的参数值将基于此基准按比例缩放

    def get_adaptive_params(self, image_shape):
        """
        根据图像分辨率计算自适应参数。

        【核心数学原理】

        本方法需要解决的根本问题是：同一叶片在不同分辨率下的像素数量差异巨大。
        例如一片叶子在 QVGA (320×240) 中占 10000 像素，在 UXGA (1600×1200) 中占 250000 像素。
        如果使用固定的像素阈值，要么在低分辨率下漏检，要么在高分辨率下误检。

        核心思路：
          - 面积比例阈值（MIN_AREA_RATIO）：高分辨率下叶片在画面中占比不变，但像素数暴增，
            需要降低比例阈值以保留叶片轮廓。采用对数衰减而非线性衰减，因为：
            线性衰减在高倍率下会趋近于 0（导致碎片也被保留），
            而对数衰减 log(x+1) 增长缓慢，能在高倍率下保持合理的下限。
          - 绝对像素阈值（DISEASE_MIN_AREA / PEST_MIN_AREA）：病斑/虫害的最小面积按像素数等比缩放，
            因为它们是物理尺寸的映射，面积与像素数成正比。
          - 形态学核大小：随分辨率适度增大，但限制范围避免过度处理。
            使用 sqrt_scale（线性尺寸缩放比）而非 scale_factor（面积缩放比），
            因为核大小是二维的边长，应该与线性尺寸成比例。

        :param image_shape: 图像形状 (height, width, channels)
        :return: 包含所有自适应参数的字典
        """
        # 提取图像的宽度和高度，计算当前总像素数
        height, width = image_shape[:2]
        current_pixels = height * width
        # 基准像素总数：320 × 240 = 76800（QVGA 分辨率）
        base_pixels = self.base_resolution[0] * self.base_resolution[1]  # 76800

        # --- 非自适应模式：返回固定基准值 ---
        # 当分辨率固定不变时（如始终使用 QVGA），无需动态计算，直接返回经验值
        if not self.resolution_adaptive:
            return {
                'MIN_AREA_RATIO': 0.03,       # 叶片轮廓最小面积占全图 3%
                'DISEASE_MIN_AREA': 80,       # 病斑最小 80 像素
                'PEST_MIN_AREA': 80,          # 虫害最小 80 像素
                'MORPH_KERNEL_SIZE': 3,       # 形态学核 3×3
                'CLOSE_KERNEL_SIZE': 7,       # 闭运算核 7×7
                'DILATE_ITERATIONS': 1,       # 膨胀 1 次
                'MEDIAN_KERNEL_SIZE': 5,      # 中值滤波核 5×5
            }

        # --- 计算缩放因子 ---

        # 像素总量缩放比 = 当前像素数 / 基准像素数
        # 例如：UXGA (1600×1200=1920000) / QVGA (76800) = 25 倍
        # 该值反映的是面积的放大倍数
        scale_factor = current_pixels / base_pixels

        # 线性尺寸缩放比 = 面积缩放比的平方根
        # 为什么需要 sqrt_scale：
        #   面积缩放是二维的（长×宽各放大 sqrt(scale_factor) 倍），
        #   而形态学核大小、膨胀距离等是一维参数（边长），应该与线性尺寸成比例。
        #   例如：面积放大 25 倍 → 线性放大 5 倍 → 核大小应乘以 5 而非 25
        #   如果用 scale_factor 直接乘核大小，UXGA 下核会变成 3×25=75，显然过大。
        sqrt_scale = np.sqrt(scale_factor)

        # --- 自适应参数计算 ---

        # 【叶片轮廓最小面积比例 - 对数衰减】
        # 数学公式：min_area_ratio = 0.03 / log(scale_factor + 1)
        #
        # 为什么用对数衰减而不是线性衰减（0.03 / scale_factor）：
        #   - 线性衰减：UXGA (25x) → 0.03/25 = 0.0012，太小了，几乎任何碎片都能通过
        #   - 对数衰减：UXGA (25x) → 0.03/log(26) ≈ 0.03/3.26 ≈ 0.009，仍然合理
        #   对数函数 log(x+1) 在 x 较小时近似线性，在 x 较大时增长缓慢，
        #   这使得阈值在高分辨率下不会过快趋近于零。
        #
        # +1 的原因：当 scale_factor=1（基准分辨率）时，log(1+1)=log(2)≈0.69，
        #   此时 min_area_ratio ≈ 0.043，略高于基准值 0.03，这是可接受的偏差。
        #
        # 最终用 max/min 限制在 [0.005, 0.05] 范围内，防止极端值
        min_area_ratio = 0.03 / np.log(scale_factor + 1)
        min_area_ratio = max(0.005, min(0.05, min_area_ratio))

        # 【病斑/虫害最小像素面积 - 等比缩放】
        # 直接按面积缩放比 scale_factor 等比放大基准值 80 像素
        # 例如：UXGA (25x) → 80 × 25 = 2000 像素，这是合理的，
        # 因为同样大小的病斑在 UXGA 中确实占据约 2000 像素。
        # max(50, ...) 确保即使在极低分辨率下也不会低于 50 像素（避免噪声被识别为病斑）
        disease_min_area = max(50, int(80 * scale_factor))
        pest_min_area = max(50, int(80 * scale_factor))

        # 【形态学核大小 - 随分辨率适度增大】
        # 基准 3×3，乘以线性缩放比 sqrt_scale，限制在 [3, 7] 范围内。
        #
        # 为什么必须为奇数：
        #   OpenCV 的形态学操作（erode/dilate/morphologyEx）要求核尺寸为奇数，
        #   因为奇数尺寸的核有明确的中心像素（例如 3×3 的中心是 [1,1]），
        #   偶数尺寸的核没有对称中心，会导致锚点偏移和处理结果不对称。
        #   如果计算出的核大小为偶数，则 +1 使其变为奇数。
        morph_kernel = max(3, min(7, int(3 * sqrt_scale)))
        if morph_kernel % 2 == 0:
            morph_kernel += 1

        # 【闭运算核大小 - 用于填充高光空洞】
        # 闭运算核需要比形态学核更大，因为它需要跨越整个高光区域进行填充。
        # 典型的高光区域在 QVGA 下约 5~7 像素宽，高分辨率下等比放大。
        # 基准 5×5，乘以 sqrt_scale，限制在 [5, 13] 范围内，同样必须为奇数。
        close_kernel = max(5, min(13, int(5 * sqrt_scale)))
        if close_kernel % 2 == 0:
            close_kernel += 1

        # 【膨胀迭代次数 - 限制 1~2】
        # 1 次膨胀足以连接大部分相邻叶片碎片；
        # 仅在高分辨率下（线性缩放比 >= 2，即面积 >= 4 倍基准）才增加到 2 次，
        # 因为高分辨率下叶片边缘的断裂距离更宽。
        dilate_iter = max(1, min(2, int(1 * sqrt_scale)))

        # 【中值滤波核大小 - 限制 3~7 且必须为奇数】
        # 中值滤波用于预处理去噪，核越大去噪越强但也会模糊细节。
        # 与形态学核相同的缩放逻辑和奇数约束。
        median_kernel = max(3, min(7, int(3 * sqrt_scale)))
        if median_kernel % 2 == 0:
            median_kernel += 1

        # 返回所有自适应参数组成的字典
        return {
            'MIN_AREA_RATIO': min_area_ratio,        # 叶片轮廓最小面积比例
            'DISEASE_MIN_AREA': disease_min_area,    # 病斑最小像素面积
            'PEST_MIN_AREA': pest_min_area,          # 虫害最小像素面积
            'MORPH_KERNEL_SIZE': morph_kernel,       # 形态学核边长（奇数）
            'CLOSE_KERNEL_SIZE': close_kernel,       # 闭运算核边长（奇数）
            'DILATE_ITERATIONS': dilate_iter,        # 膨胀迭代次数
            'MEDIAN_KERNEL_SIZE': median_kernel,     # 中值滤波核边长（奇数）
        }

    def to_dict(self):
        """
        将配置对象的所有参数转换为字典格式。
        用途：在 Web API 中将参数序列化为 JSON 传输给前端，
        或在日志中记录当前使用的参数配置。
        :return: 参数字典，键为参数名，值为参数值
        """
        return {k: v for k, v in vars(self).items()}

    def from_dict(self, data):
        """
        从字典加载参数并覆盖当前配置。
        用途：Web API 接收前端传来的参数调整后，动态更新检测配置。
        只更新字典中存在的且配置对象中已有的属性，忽略无效键名，
        防止前端传入非法参数名导致程序异常。
        :param data: 参数字典，键值对对应配置属性名和值
        """
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)


# 默认配置实例：模块加载时创建一次，供 detect_and_annotate() 等函数在未传入 config 时使用
DEFAULT_CONFIG = DetectionConfig()


# ====================================================================================
# 叶片分割函数
# ====================================================================================


def segment_leaves(image, config):
    """
    简化版叶片分割 - 绿色通道优势 AND 饱和度筛选 + 形态学填充高光 + 分辨率自适应

    本函数是整个检测流水线的核心前置步骤，负责从复杂背景中精确提取叶片区域。
    输出为二值掩码（leaf_mask），其中白色（255）为叶片像素，黑色（0）为背景。

    完整处理流程（8步）：
      第1步 中值滤波去噪     → 消除 JPEG 压缩伪影和传感器噪声，为后续颜色判断提供干净数据
      第2步 绿色通道优势     → G - max(R,B) > GREEN_ADV，识别"比红/蓝都更绿"的像素
      第3步 饱和度下限       → S >= SAT_MIN，过滤键盘等低饱和度背景
      第4步 条件1 AND 条件2  → 纯 AND 逻辑合并，不引入额外误判
      第5步 形态学闭运算     → 填充叶片上的高光空洞（反光区域的绿色被白色光冲淡）
      第6步 形态学开运算     → 去除背景中的孤立噪点
      第7步 适度膨胀         → 连接断开的叶片部分（叶脉或阴影可能导致断裂）
      第8步 轮廓面积过滤     → 去除残余碎片（小于面积阈值的区域不是叶片）

    :param image: BGR 格式的输入图像（OpenCV 默认读取格式）
    :param config: DetectionConfig 实例，包含所有检测参数
    :return: (leaf_mask, green_ratio) 二元组
             - leaf_mask: uint8 二值掩码，255=叶片，0=背景
             - green_ratio: 叶片像素占全图比例（0.0~1.0），用于后续分级判断
    """
    # 获取图像尺寸信息，用于后续计算面积比例
    height, width = image.shape[:2]
    total_pixels = height * width

    # 根据当前图像分辨率获取自适应参数（核大小、面积阈值等）
    params = config.get_adaptive_params(image.shape)

    # ---- 第1步：中值滤波去噪 ----
    # 中值滤波（medianBlur）相比高斯模糊的优势：
    #   - 能有效去除椒盐噪声（JPEG 压缩产生的亮点/暗点）
    #   - 同时保留边缘清晰度（不会让叶片边缘变模糊）
    #   - 这对后续的颜色阈值分割非常重要，因为模糊的边缘会产生过渡色像素
    median_k = params['MEDIAN_KERNEL_SIZE']
    image_blurred = cv2.medianBlur(image, median_k)

    # ---- 第2步：绿色通道优势 ----
    # 【为什么需要 int16 而不是 uint8】
    # OpenCV 读取的图像是 uint8 类型（无符号 8 位整数，范围 0~255）。
    # 在 uint8 下做减法会发生"下溢回绕"：例如 G=100, R=200 时，
    #   uint8: 100 - 200 = -100 → 回绕为 156（错误！应该是负数）
    #   int16: 100 - 200 = -100（正确，负数表示该像素不是绿色优势）
    # 转换为 int16（有符号 16 位整数，范围 -32768~32767）可安全进行减法运算。
    b = image_blurred[:, :, 0].astype(np.int16)  # B 通道（蓝色）
    g = image_blurred[:, :, 1].astype(np.int16)  # G 通道（绿色）
    r = image_blurred[:, :, 2].astype(np.int16)  # R 通道（红色）

    # 计算每个像素在红色和蓝色通道中的最大值
    # 绿色优势 = G - max(R, B)，只有当绿色比红色和蓝色都强时，差值才为正
    # 这比简单的 G > R 或 G > B 更严格，避免了橙色（G>R 但 G<B）等误判
    max_rb = np.maximum(r, b)
    green_diff = g - max_rb
    # 差值大于 GREEN_ADV（默认12）的像素被标记为"绿色优势像素"
    green_adv_mask = green_diff > config.GREEN_ADV

    # ---- 第3步：饱和度下限 ----
    # 将图像从 BGR 色彩空间转换到 HSV（色相-饱和度-明度）空间
    # HSV 更适合颜色筛选，因为饱和度 S 独立于亮度 V
    hsv = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]  # 提取 S 通道（饱和度，0=灰度，255=纯色）
    # 饱和度 >= SAT_MIN（默认25）的像素被标记为"足够饱和"
    # 过滤效果：键盘、水泥地面、阴影区域等低饱和度物体被排除
    sat_mask = s_channel >= config.SAT_MIN

    # ---- 第4步：AND 合并 ----
    # 【为什么用 AND 而不是 OR】
    # AND 逻辑：只有同时满足"绿色优势"和"足够饱和度"才认定为叶片。
    #   - 只用绿色优势：可能误判绿色桌面、绿色布料等非叶片绿色物体
    #   - 只用饱和度：所有高饱和度物体（红色花盆、蓝色标签）都会被误判
    #   - AND 合并：必须是"既绿又有色彩"的像素，大幅提高精确率
    # OR 逻辑虽然能提高召回率（检出更多叶片），但会显著增加误判，
    # 而本系统宁可漏检一些边缘叶片，也不要误判背景为叶片（影响后续病斑统计）。
    # & 为 NumPy 按位与运算，将两个布尔掩码取交集
    leaf_mask = (green_adv_mask & sat_mask).astype(np.uint8) * 255

    # ---- 第5步：形态学闭运算 - 填充高光空洞 ----
    # 【为什么闭运算能填充高光空洞】
    # 叶片表面常有反光/高光点，这些区域的绿色被白色光"冲淡"，
    # 导致在第2步中 G-max(R,B) 差值变小而被排除，形成叶片掩码中的"空洞"。
    #
    # 形态学闭运算 = 先膨胀后腐蚀（dilate → erode）：
    #   - 膨胀阶段：叶片区域向外扩张，"吞并"周围的小空洞
    #   - 腐蚀阶段：叶片区域收缩回原始大小，但已被填充的空洞不会恢复
    # 最终效果：叶片上的小空洞被填平，而叶片整体轮廓基本不变。
    #
    # 使用椭圆形核（MORPH_ELLIPSE）而非矩形核，
    # 因为叶片边缘多为曲线，椭圆核能更好地贴合自然形状。
    close_k = params['CLOSE_KERNEL_SIZE']
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # ---- 第6步：形态学开运算 - 去除背景噪点 ----
    # 形态学开运算 = 先腐蚀后膨胀（erode → dilate）：
    #   - 腐蚀阶段：所有区域缩小，小的噪点（几个像素大小）被完全消除
    #   - 膨胀阶段：存活下来的大区域恢复原始大小
    # 最终效果：小于核尺寸的孤立噪点被去除，而叶片等大目标不受影响。
    # 注意：开运算必须在闭运算之后执行，否则闭运算填充的空洞会被开运算重新打开。
    morph_k = params['MORPH_KERNEL_SIZE']
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    # ---- 第7步：适度膨胀 - 连接断开的叶片部分 ----
    # 叶脉、阴影或高光带可能将一片完整叶片"切断"为多个碎片。
    # 适度膨胀（1~2次）使相邻碎片边缘扩展，从而重新连接为一个整体。
    # 过度膨胀（3次以上）会导致叶片轮廓严重变形，因此通过自适应参数限制迭代次数。
    dilate_iter = params['DILATE_ITERATIONS']
    leaf_mask = cv2.dilate(leaf_mask, morph_kernel, iterations=dilate_iter)

    # ---- 第8步：轮廓面积过滤 - 去除残余碎片 ----
    # 经过前面所有步骤后，仍可能存在一些不属于叶片的小区域（如远处的绿色背景碎片）。
    # 通过提取所有外部轮廓，计算每个轮廓的面积，
    # 只保留面积 >= MIN_AREA_RATIO（占全图一定比例以上）的轮廓作为真正的叶片。
    # RETR_EXTERNAL: 只检测最外层轮廓（忽略内部孔洞）
    # CHAIN_APPROX_SIMPLE: 压缩水平/垂直/对角线方向的冗余点，节省内存
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 最小面积 = 全图像素数 × 最小面积比例（自适应值）
    min_area = total_pixels * params['MIN_AREA_RATIO']

    # 创建空白掩码，只将通过面积筛选的轮廓绘制上去
    filtered_mask = np.zeros_like(leaf_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            # 绘制轮廓并填充内部（thickness=-1 表示填充）
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    leaf_mask = filtered_mask

    # 计算绿色叶片像素占全图的比例，用于后续分级判断（叶片枯萎程度评估）
    green_pixels = cv2.countNonZero(leaf_mask)
    green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0

    return leaf_mask, green_ratio


# ====================================================================================
# 病斑检测函数
# ====================================================================================


def detect_disease(image, leaf_mask, config):
    """
    病斑检测 - 在叶片掩码范围内识别棕色和黄色病变区域。

    农作物病害在叶片上的典型表现：
      - 棕色病斑：早期坏死、褐斑病等，在 HSV 空间中 H 值接近红色端（0~15）
      - 黄色病斑：黄化病、褪绿病等，在 HSV 空间中 H 值在黄色区间（18~32）
    两类病斑的共同特征：饱和度适中（S >= 50），不会是纯灰或纯白区域。

    检测流程：
      1. 转换到 HSV 空间，分别提取棕色和黄色掩码
      2. 合并两类掩码，用叶片掩码限制检测范围
      3. 提取轮廓，按最小面积过滤噪点
      4. 对每个轮廓计算质心，验证质心确实在叶片内部

    :param image: BGR 格式的原始输入图像
    :param leaf_mask: 由 segment_leaves() 生成的叶片二值掩码
    :param config: DetectionConfig 实例
    :return: (disease_mask, disease_contours)
             - disease_mask: 病斑区域的二值掩码
             - disease_contours: 通过筛选的病斑轮廓列表
    """
    # 获取分辨率自适应参数（主要用于最小面积阈值）
    params = config.get_adaptive_params(image.shape)

    # 将原始图像从 BGR 转换到 HSV 色彩空间
    # 注意：这里使用原始图像而非预处理后的图像，因为病斑的颜色特征需要原始像素值
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)  # 分离 H（色相）、S（饱和度）、V（明度）三个通道

    # 【棕色病斑检测 - HSV 范围说明】
    # 在 OpenCV 的 HSV 空间中，H 的范围是 0~180（对应 0°~360° 的一半）：
    #   H=0   对应 0°（正红色）
    #   H=15  对应 30°（橙红色）
    #   H=18  对应 36°（橙色到黄色的过渡）
    # 棕色在色彩学上是低明度的橙红色，因此在 HSV 中表现为低 H 值 + 中等 S 值。
    # S >= 50 排除了低饱和度的灰色/白色区域（如高光反光点）
    brown_mask = ((h >= config.BROWN_H_MIN) & (h <= config.BROWN_H_MAX)
                  & (s >= config.BROWN_S_MIN))

    # 【黄色病斑检测 - HSV 范围说明】
    # 黄色在 HSV 中的典型范围：
    #   H=18  对应 36°（橙黄色）
    #   H=32  对应 64°（黄绿色过渡区）
    # 注意 H=15~18 之间有一个"间隙"，这是橙色区间，
    # 既不属于棕色也不属于黄色，避免边界区域的重复计数。
    # S >= 50 与棕色相同的饱和度下限
    yellow_mask = ((h >= config.YELLOW_H_MIN) & (h <= config.YELLOW_H_MAX)
                   & (s >= config.YELLOW_S_MIN))

    # 合并棕色和黄色病斑掩码（取并集）
    # 使用 | （按位或）将两类病斑合为一个掩码
    disease_mask = (brown_mask | yellow_mask).astype(np.uint8) * 255

    # 用叶片掩码限制检测范围：只保留叶片区域内的病斑
    # 目的：排除背景中可能存在的棕/黄色物体（如泥土、木桌）
    # bitwise_and 配合 mask 参数，等效于 disease_mask & leaf_mask
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

    # 提取病斑区域的外部轮廓
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 【轮廓过滤 - 面积 + 质心双重验证】
    disease_min_area = params['DISEASE_MIN_AREA']  # 分辨率自适应的最小面积阈值
    filtered_contours = []
    h_bound, w_bound = leaf_mask.shape[:2]  # 获取掩码的行列数（高度和宽度）

    for cnt in contours:
        # 第一层过滤：面积必须大于最小阈值（排除噪声碎片）
        if cv2.contourArea(cnt) >= disease_min_area:
            # 计算轮廓的矩（moments），用于求质心坐标
            # m00 = 轮廓面积（像素数），m10/m01 = 一阶矩
            M = cv2.moments(cnt)
            if M['m00'] > 0:  # 面积为正才计算质心（排除退化轮廓）
                # 质心坐标 = (m10/m00, m01/m00)，即面积加权平均位置
                cx = int(M['m10'] / M['m00'])  # 质心 x 坐标（列号）
                cy = int(M['m01'] / M['m00'])  # 质心 y 坐标（行号）
                # 【边界保护 + 质心验证】
                # 为什么需要检查质心：
                #   1. 边界保护：质心坐标可能因取整而超出图像边界（如轮廓贴边时），
                #      直接用 leaf_mask[cy, cx] 访问会导致数组越界异常。
                #   2. 质心在叶片内验证：虽然病斑掩码已经用叶片掩码过滤过，
                #      但轮廓检测可能生成跨越叶片边缘的轮廓（如病斑恰好位于叶缘），
                #      此时轮廓面积虽然大，但质心可能在叶片外部。
                #      只有质心确实在叶片内的轮廓才被认定为有效病斑。
                if 0 <= cy < h_bound and 0 <= cx < w_bound:
                    if leaf_mask[cy, cx] == 255:  # 质心位于叶片区域内
                        filtered_contours.append(cnt)

    return disease_mask, filtered_contours


# ====================================================================================
# 虫害检测函数
# ====================================================================================


def detect_pests(image, leaf_mask, config):
    """
    虫害检测 - 在叶片掩码范围内识别白色虫害痕迹（如白粉虱、蚜虫蜕皮、虫卵等）。

    【白色检测逻辑说明】
    虫害在叶片上的典型视觉表现是白色小点或白色粉末状区域。
    在 HSV 色彩空间中，白色的特征是：
      - 饱和度极低（S 接近 0）：白色是"无色"的，没有明显的色调
      - 亮度极高（V 接近 255）：白色反射所有波长的光，亮度最高
    因此检测条件为：S <= PEST_S_MAX（15）且 V >= PEST_V_MIN（220）
    这两个阈值设得非常严格（饱和度极低 + 亮度极高），目的是：
      - 避免将浅色叶片（浅绿色 S 约 30~60）误判为虫害
      - 避免将高光反光（虽然亮但 S 可能为 20~30）误判为虫害
    只有接近纯白的亮点才会被识别。

    :param image: BGR 格式的原始输入图像
    :param leaf_mask: 由 segment_leaves() 生成的叶片二值掩码
    :param config: DetectionConfig 实例
    :return: (pest_mask, pest_contours)
             - pest_mask: 虫害区域的二值掩码
             - pest_contours: 通过筛选的虫害轮廓列表
    """
    # 获取分辨率自适应参数（主要用于最小面积阈值和形态学核大小）
    params = config.get_adaptive_params(image.shape)

    # 转换到 HSV 空间进行颜色筛选
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 白色虫害点检测：饱和度极低 AND 亮度极高
    # 两个条件缺一不可：
    #   - 只要求 S<=15：暗灰色也满足（如阴影中的灰尘），但它们不是虫害
    #   - 只要求 V>=220：亮黄色也满足（如阳光直射的叶片），但它们不是虫害
    #   - 同时要求：只有"既亮又无色"的纯白点才被识别
    pest_mask = ((s <= config.PEST_S_MAX) & (v >= config.PEST_V_MIN)).astype(np.uint8) * 255

    # 用叶片掩码限制范围：只检测叶片上的虫害，忽略背景中的白色物体
    pest_mask = cv2.bitwise_and(pest_mask, pest_mask, mask=leaf_mask)

    # 形态学开运算去除噪点
    # 虫害白点通常有一定面积（几十到几百像素），而图像噪声产生的白点通常只有 1~3 像素。
    # 开运算（先腐蚀后膨胀）可以消除小于核尺寸的微小噪点，保留真正的虫害区域。
    morph_k = params['MORPH_KERNEL_SIZE']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    pest_mask = cv2.morphologyEx(pest_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 提取虫害区域轮廓并进行面积 + 质心双重过滤（逻辑与 detect_disease 相同）
    contours, _ = cv2.findContours(pest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pest_min_area = params['PEST_MIN_AREA']  # 分辨率自适应的最小虫害面积
    filtered_contours = []
    h_bound, w_bound = leaf_mask.shape[:2]

    for cnt in contours:
        if cv2.contourArea(cnt) >= pest_min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # 边界保护：防止数组越界；质心验证：确保虫害点确实在叶片上
                if 0 <= cy < h_bound and 0 <= cx < w_bound:
                    if leaf_mask[cy, cx] == 255:
                        filtered_contours.append(cnt)

    return pest_mask, filtered_contours


# ====================================================================================
# 严重程度分级函数
# ====================================================================================


def classify_level(disease_count, disease_ratio, pest_count, pest_ratio, green_ratio, config):
    """
    分级判断 - 基于"数量 OR 占比"双阈值体系确定病虫害严重程度。

    【双阈值系统说明】
    对病斑和虫害分别设定了两种类型的阈值：
      1. 数量阈值（count）：检测到的独立病斑/虫害个数
      2. 面积占比阈值（ratio）：病斑/虫害像素占全图总面积的比例

    采用 OR 逻辑：任一指标超过阈值即触发对应等级。
    这样设计的原因：
      - "少而大"模式：可能只有 2 个病斑但面积巨大（占比 10%），数量阈值无法捕获
      - "多而小"模式：可能有 10 个微小病斑但总面积极小，占比阈值无法捕获
    双阈值 OR 逻辑能同时覆盖两种极端情况。

    【等级判定顺序】
    从低到高依次判断：绿色不足 → 病斑 → 虫害
    使用 max(level_code, new_code) 机制保留最高等级：
      如果绿色不足已经是"警告"（code=2），但病斑判定为"严重"（code=3），
      最终结果为"严重"（取最高等级，不做降级）。

    等级定义：
      - code 0: 正常    → 各项指标均在安全范围内
      - code 1: 注意    → 出现轻微病虫害迹象，需要关注
      - code 2: 警告    → 病虫害较明显，建议采取措施
      - code 3: 严重    → 病虫害严重，需要立即处理

    :param disease_count: 检测到的病斑数量
    :param disease_ratio: 病斑面积占全图比例（0.0~1.0）
    :param pest_count: 检测到的虫害白点数量
    :param pest_ratio: 虫害面积占全图比例（0.0~1.0）
    :param green_ratio: 绿色叶片占全图比例（0.0~1.0）
    :param config: DetectionConfig 实例
    :return: (level_code, level_name) 二元组
    """
    level_code = 0       # 等级代码：0=正常，1=注意，2=警告，3=严重
    level_name = "正常"

    # ---- 第一维度：叶片整体健康程度（绿色占比） ----
    # 绿色占比反映叶片的整体存活/健康状态：
    #   - 占比 < 10%：叶片严重枯萎或几乎无健康叶片 → 警告
    #   - 占比 < 20%：叶片覆盖不足或大面积褪绿 → 注意
    # 注意：绿色占比低不一定意味着病虫害，也可能是拍摄角度问题，
    # 因此这里的等级不会超过"警告"（code=2），避免过度报警。
    if green_ratio < config.GREEN_RATIO_WARNING:
        level_code = 2
        level_name = "警告"
    elif green_ratio < config.GREEN_RATIO_NOTICE:
        level_code = 1
        level_name = "注意"

    # ---- 第二维度：病斑严重程度 ----
    # 使用 elif 链从严重到注意依次判断，一旦命中即停止（取命中的最高等级）。
    # 每个等级内部使用 OR 逻辑：数量 >= 阈值 或 占比 > 阈值 均可触发。
    # 使用 max(level_code, ...) 确保不会因后续判断覆盖已有的更高严重等级。
    if disease_count >= config.SERIOUS_DISEASE_COUNT or disease_ratio > config.SERIOUS_DISEASE_RATIO:
        level_code = max(level_code, 3)
        level_name = "严重"
    elif disease_count >= config.WARNING_DISEASE_COUNT or disease_ratio > config.WARNING_DISEASE_RATIO:
        level_code = max(level_code, 2)
        level_name = "警告"
    elif disease_count >= config.NOTICE_DISEASE_COUNT or disease_ratio > config.NOTICE_DISEASE_RATIO:
        level_code = max(level_code, 1)
        level_name = "注意"

    # ---- 第三维度：虫害严重程度 ----
    # 逻辑与病斑相同：数量 OR 占比，从高到低依次判断。
    # 注意虫害的"注意级"只有数量阈值（NOTICE_PEST_COUNT），没有占比阈值，
    # 因为虫害白点通常面积很小，在整体占比上几乎不可见，
    # 用数量计数比面积占比更敏感和实用。
    if pest_count >= config.SERIOUS_PEST_COUNT or pest_ratio > config.SERIOUS_PEST_RATIO:
        level_code = max(level_code, 3)
        level_name = "严重"
    elif pest_count >= config.WARNING_PEST_COUNT or pest_ratio > config.WARNING_PEST_RATIO:
        level_code = max(level_code, 2)
        level_name = "警告"
    elif pest_count >= config.NOTICE_PEST_COUNT:
        level_code = max(level_code, 1)
        level_name = "注意"

    return level_code, level_name


# ====================================================================================
# 完整检测与标注函数（主入口）
# ====================================================================================


def detect_and_annotate(image, config=None):
    """
    完整检测流程 - 串联所有检测步骤并在原始图像上绘制可视化标注。

    本函数是整个模块的主入口，执行以下完整流水线：
      步骤1: 叶片分割 → 获取叶片掩码和绿色占比
      步骤2: 病斑检测 → 在叶片范围内识别棕/黄色病变区域
      步骤3: 虫害检测 → 在叶片范围内识别白色虫害痕迹
      步骤4: 面积占比计算 → 统计病斑和虫害的像素面积占比
      步骤5: 分级判断 → 根据双阈值体系确定严重程度等级
      步骤6: 可视化标注 → 在原始图像上绘制轮廓、边框和文字信息面板

    :param image: BGR 格式的输入图像
    :param config: DetectionConfig 实例，默认为 DEFAULT_CONFIG（模块级默认配置）
    :return: (result_dict, annotated_image, masks_dict) 三元组
             - result_dict: 包含检测结果的字典（等级、数量、占比等）
             - annotated_image: 绘制了标注的 BGR 图像
             - masks_dict: 包含各类二值掩码的字典（用于调试或进一步分析）
    """
    # 如果未传入配置，使用模块级默认配置
    if config is None:
        config = DEFAULT_CONFIG

    # 获取图像尺寸信息
    height, width = image.shape[:2]
    total_pixels = height * width

    # ---- 步骤1：叶片分割 ----
    # 从复杂背景中提取叶片区域，输出二值掩码和绿色占比
    leaf_mask, green_ratio = segment_leaves(image, config)

    # ---- 步骤2：病斑检测 ----
    # 在叶片掩码范围内识别棕色和黄色病变区域
    disease_mask, disease_contours = detect_disease(image, leaf_mask, config)

    # ---- 步骤3：虫害检测 ----
    # 在叶片掩码范围内识别白色虫害痕迹
    pest_mask, pest_contours = detect_pests(image, leaf_mask, config)

    # ---- 步骤4：计算面积占比 ----
    # 通过累加所有通过筛选的轮廓面积来统计病斑/虫害的总像素数
    # 注意：使用 contourArea（轮廓面积）而非 countNonZero（掩码非零像素），
    # 因为轮廓面积更精确（考虑了轮廓形状的亚像素精度）
    disease_pixels = sum(cv2.contourArea(cnt) for cnt in disease_contours)
    pest_pixels = sum(cv2.contourArea(cnt) for cnt in pest_contours)
    # 面积占比 = 病虫害像素数 / 全图总像素数（不是叶片面积，因为等级阈值是基于全图校准的）
    disease_ratio = disease_pixels / total_pixels if total_pixels > 0 else 0.0
    pest_ratio = pest_pixels / total_pixels if total_pixels > 0 else 0.0

    # ---- 步骤5：分级判断 ----
    # 将病斑数量/占比、虫害数量/占比、绿色占比传入分级函数，获取严重程度等级
    level_code, level_name = classify_level(
        len(disease_contours), disease_ratio,
        len(pest_contours), pest_ratio,
        green_ratio, config
    )

    # ---- 步骤6：绘制可视化标注图 ----
    # 在原始图像的副本上绘制标注（不修改原图）
    annotated = image.copy()

    # 【绘制叶片轮廓 - 绿色线条】
    # 提取叶片掩码的外部轮廓并用绿色（BGR: 0,255,0）描边，线宽2像素
    # 用于直观展示系统识别到的叶片区域边界
    leaf_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, leaf_contours, -1, (0, 255, 0), 2)

    # 【绘制病斑边框 - 红色矩形】
    # 对每个病斑轮廓计算最小外接矩形（boundingRect），用红色（BGR: 0,0,255）绘制
    # 矩形框直观标出每个病斑的位置和大小
    for cnt in disease_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 【绘制虫害边框 - 蓝色矩形】
    # 与病斑相同的矩形框逻辑，但用蓝色（BGR: 255,0,0）区分
    for cnt in pest_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # ---- 绘制文字信息面板 ----
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用简单线段字体（不支持中文，因此标签用英文）
    img_height, img_width = annotated.shape[:2]

    # 【自适应字体大小】
    # 根据图像宽度动态调整字体缩放比例，确保在不同分辨率下文字都清晰可读：
    #   - 低分辨率（320px宽）：font_scale ≈ 0.27 → 限制到最小值 0.4
    #   - 中分辨率（640px宽）：font_scale ≈ 0.53 → 使用计算值
    #   - 高分辨率（1600px宽）：font_scale ≈ 1.33 → 限制到最大值 0.6
    # 行高、内边距、边框粗细也按相同比例缩放
    font_scale = max(0.4, min(0.6, img_width / 1200))
    line_height = int(22 * font_scale)
    padding = int(8 * font_scale)
    border_thickness = int(2 * font_scale)

    # 【等级对应的颜色方案】
    # 每个等级有三种颜色：背景色（bg）、边框色（border）、文字色（text）
    # 颜色遵循交通灯隐喻：绿色=正常、橙色=注意、红色=警告/严重
    level_colors = {
        "正常": {"bg": (30, 150, 30), "border": (0, 255, 0), "text": (255, 255, 255)},    # 绿色系
        "注意": {"bg": (180, 120, 30), "border": (255, 165, 0), "text": (255, 255, 255)},  # 橙色系
        "警告": {"bg": (180, 60, 30), "border": (255, 69, 0), "text": (255, 255, 255)},    # 红橙色系
        "严重": {"bg": (150, 30, 30), "border": (255, 0, 0), "text": (255, 255, 255)},     # 红色系
    }
    colors = level_colors.get(level_name, level_colors["正常"])

    # 【构建文字行及颜色】
    # 每行文字包含标签和数值，当数值超过注意阈值时，该行文字变为红色以示警告
    disease_count = len(disease_contours)
    pest_count = len(pest_contours)

    texts = []            # 文字内容列表
    colors_per_line = []  # 每行对应的颜色列表

    # 第1行：严重程度等级（始终使用等级对应的默认文字色）
    texts.append("Level: {}".format(level_name))
    colors_per_line.append(colors["text"])

    # 第2行：病斑统计（达到注意阈值时变红）
    if disease_count >= config.NOTICE_DISEASE_COUNT or disease_ratio >= config.NOTICE_DISEASE_RATIO:
        texts.append("Disease: {} ({:.1%})".format(disease_count, disease_ratio))
        colors_per_line.append((0, 0, 255))  # 红色文字 = 超标警告
    else:
        texts.append("Disease: {} ({:.1%})".format(disease_count, disease_ratio))
        colors_per_line.append(colors["text"])  # 默认文字色 = 正常

    # 第3行：虫害统计（达到注意阈值时变红）
    if pest_count >= config.NOTICE_PEST_COUNT or pest_ratio >= config.WARNING_PEST_RATIO:
        texts.append("Pest: {} ({:.1%})".format(pest_count, pest_ratio))
        colors_per_line.append((0, 0, 255))  # 红色文字 = 超标警告
    else:
        texts.append("Pest: {} ({:.1%})".format(pest_count, pest_ratio))
        colors_per_line.append(colors["text"])  # 默认文字色 = 正常

    # 第4行：绿色占比（低于注意阈值时变红，表示叶片不足）
    if green_ratio < config.GREEN_RATIO_NOTICE:
        texts.append("Green: {:.1%}".format(green_ratio))
        colors_per_line.append((0, 0, 255))  # 红色文字 = 叶片不足警告
    else:
        texts.append("Green: {:.1%}".format(green_ratio))
        colors_per_line.append(colors["text"])  # 默认文字色 = 正常

    # 【计算文字面板尺寸】
    # 遍历所有文字行，找到最宽的一行作为面板宽度基准
    max_text_width = 0
    for text in texts:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
        max_text_width = max(max_text_width, tw)
    box_width = max_text_width + padding * 2         # 面板宽度 = 最宽文字 + 左右内边距
    box_height = len(texts) * line_height + padding * 2  # 面板高度 = 行数 × 行高 + 上下内边距

    # 【在图像左上角绘制信息面板】
    # 先绘制半透明背景矩形（填充），再绘制边框矩形
    x_start = padding  # 面板左上角 x 坐标
    y_start = padding  # 面板左上角 y 坐标

    # 绘制填充背景（thickness=-1 表示实心填充）
    cv2.rectangle(annotated, (x_start, y_start),
                  (x_start + box_width, y_start + box_height),
                  colors["bg"], -1)
    # 绘制边框线条
    cv2.rectangle(annotated, (x_start, y_start),
                  (x_start + box_width, y_start + box_height),
                  colors["border"], border_thickness)

    # 逐行绘制文字（putText 的 y 坐标是文字基线位置，所以需要 +line_height 留出上方空间）
    for i, (text, text_color) in enumerate(zip(texts, colors_per_line)):
        y = y_start + padding + line_height + i * line_height
        cv2.putText(annotated, text, (x_start + padding, y),
                    font, font_scale, text_color, 2)

    # ---- 封装检测结果字典 ----
    # 该字典通过 Web API 返回给前端或上位机，供其展示或记录
    result = {
        "level": level_name,              # 严重程度名称（正常/注意/警告/严重）
        "level_code": level_code,         # 严重程度代码（0/1/2/3），方便程序比较
        "disease_count": len(disease_contours),  # 病斑数量
        "disease_ratio": disease_ratio,   # 病斑面积占比
        "white_count": len(pest_contours),       # 虫害白点数量（字段名保持 white 以兼容旧接口）
        "white_ratio": pest_ratio,        # 虫害面积占比
        "green_ratio": green_ratio,       # 绿色叶片占比
    }

    # 返回三元组：结果字典、标注图像、掩码字典
    # 掩码字典可用于调试（如在前端展示中间处理结果）
    return result, annotated, {
        "leaf_mask": leaf_mask,       # 叶片二值掩码
        "disease_mask": disease_mask, # 病斑二值掩码
        "pest_mask": pest_mask        # 虫害二值掩码
    }


# ====================================================================================
# 掩码可视化与图像生成辅助函数
# ====================================================================================


def visualize_mask(mask, mask_type="leaf"):
    """
    将二值掩码转换为彩色可视化图像，用于前端调试展示。

    将单通道的二值掩码（0 和 255）转换为三通道的 BGR 彩色图像，
    其中掩码中的白色区域（值为 255）被替换为对应类型的标识颜色，
    黑色区域（值为 0）保持纯黑。

    颜色约定（与标注图中的颜色一致）：
      - leaf（叶片）  → 绿色 (0, 255, 0)
      - disease（病斑）→ 红色 (0, 0, 255)
      - pest（虫害）  → 蓝色 (255, 0, 0)

    :param mask: 二值掩码（uint8，值为 0 或 255 的单通道图像）
    :param mask_type: 掩码类型标识，可选 "leaf" / "disease" / "pest"
    :return: 与掩码同尺寸的 BGR 彩色图像（黑色背景 + 彩色前景）
    """
    # 掩码类型到 BGR 颜色的映射表
    color_map = {
        "leaf": (0, 255, 0),      # 绿色 = 叶片区域
        "disease": (0, 0, 255),   # 红色 = 病斑区域
        "pest": (255, 0, 0)       # 蓝色 = 虫害区域
    }
    # 获取对应颜色，未知类型默认为绿色
    color = color_map.get(mask_type, (0, 255, 0))
    # 创建与原图同尺寸的全黑三通道图像作为底色
    result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # 将掩码中非零（>0）的像素位置设置为对应颜色
    result[mask > 0] = color
    return result


def generate_mask_image(image, params, mask_type):
    """
    根据前端传入的参数生成指定类型的掩码可视化图像，返回 JPEG 字节流。

    本函数是 Web API 的图像生成端点核心逻辑：
      前端可以通过调整参数（params）实时观察不同配置下各类掩码的效果，
      用于参数调试和可视化验证。

    支持的 mask_type 类型：
      - "leaf"     : 仅展示叶片分割结果（绿色掩码）
      - "disease"  : 展示病斑检测结果（红色掩码）
      - "pest"     : 展示虫害检测结果（蓝色掩码）
      - "annotated": 展示完整的检测标注图（包含轮廓、边框、文字面板）

    处理流程：
      1. 根据 params 字典创建 DetectionConfig 实例
      2. 根据 mask_type 调用对应的检测函数
      3. 将结果图像编码为 JPEG 格式字节流
      4. 返回字节流供 HTTP 响应直接发送

    :param image: BGR 格式的原始输入图像
    :param params: 参数字典（对应 DetectionConfig 的各属性），由前端 JSON 传入
    :param mask_type: 要生成的掩码类型（leaf/disease/pest/annotated）
    :return: JPEG 格式的图像字节流（bytes），可直接作为 HTTP 响应体返回
    """
    # 从字典创建配置实例（from_dict 会自动忽略无效键名）
    config = DetectionConfig()
    config.from_dict(params)

    if mask_type == "annotated":
        # 完整标注模式：调用主检测函数获取标注图像
        _, annotated, _ = detect_and_annotate(image, config)
        # 将标注图像编码为 JPEG 格式（imencode 返回 (retval, buf) 二元组）
        _, buf = cv2.imencode('.jpg', annotated)
        return buf.tobytes()
    else:
        # 单类掩码模式：先执行叶片分割（所有掩码都依赖叶片掩码作为范围限制）
        leaf_mask, _ = segment_leaves(image, config)

        if mask_type == "leaf":
            # 叶片掩码：直接可视化
            visualized = visualize_mask(leaf_mask, "leaf")
        elif mask_type == "disease":
            # 病斑掩码：在叶片范围内检测病斑后可视化
            disease_mask, _ = detect_disease(image, leaf_mask, config)
            visualized = visualize_mask(disease_mask, "disease")
        elif mask_type == "pest":
            # 虫害掩码：在叶片范围内检测虫害后可视化
            pest_mask, _ = detect_pests(image, leaf_mask, config)
            visualized = visualize_mask(pest_mask, "pest")
        else:
            # 未知类型：默认回退到叶片掩码
            visualized = visualize_mask(leaf_mask, "leaf")

        # 将彩色可视化图像编码为 JPEG 字节流
        _, buf = cv2.imencode('.jpg', visualized)
        return buf.tobytes()


# ====================================================================================
# 图像获取函数
# ====================================================================================


def fetch_image(url=None):
    """
    获取图像 - 支持本地测试图片和 ESP32-CAM 摄像头两种来源，带降级回退机制。

    【获取优先级与回退逻辑】
    1. 首先检查 TEST_IMAGE_PATH 是否指向一个有效的本地文件：
       - 如果是，直接读取本地图片并返回，来源标记为 "test"
       - 这用于开发调试阶段，无需连接实际硬件即可测试检测算法
    2. 如果无有效测试图片，尝试从 ESP32-CAM 摄像头获取：
       - 通过 HTTP GET 请求访问 url 参数（或全局 URL）的 /capture 接口
       - 使用流式下载（stream=True），分块读取 JPEG 数据
       - 超时设为 30 秒（ESP32-CAM 拍照+传输可能较慢）
    3. 如果摄像头获取也失败（网络不通、超时、设备离线等），返回 (None, "none")

    来源标识（source）的用途：
      - "test": 来自本地测试图片，检测结果仅用于调试
      - "camera": 来自实时摄像头，检测结果可用于实际决策
      - "none": 获取失败，调用方应提示用户检查网络或设备

    :param url: 可选，ESP32-CAM 的 capture URL。为 None 时使用模块全局 URL。
                多摄像头场景下，每个摄像头传入自己的 URL。
    :return: (image, source) 二元组
             - image: BGR 格式的 numpy 数组（成功时），或 None（失败时）
             - source: 图像来源标识（"test" / "camera" / "none"）
    """
    global TEST_IMAGE_PATH  # 引用模块级全局变量（允许外部动态修改测试路径）

    # ---- 优先尝试：本地测试图片 ----
    # 当 TEST_IMAGE_PATH 非 None 且文件确实存在时，直接读取本地图片
    # cv2.imread 返回 BGR 格式的 numpy 数组，如果文件损坏会返回 None
    if TEST_IMAGE_PATH and os.path.exists(TEST_IMAGE_PATH):
        img = cv2.imread(TEST_IMAGE_PATH)
        return img, "test"

    # ---- 降级方案：从 ESP32-CAM 摄像头获取 ----
    # 多摄像头场景: 使用传入的 url 参数; 单摄像头兼容: 使用模块全局 URL
    capture_url = url if url is not None else URL
    try:
        # stream=True: 启用流式下载，不将整个响应体缓冲到内存中
        # 这对于嵌入式设备的 HTTP 接口很重要，因为一次性传输大图可能失败
        # timeout=30: ESP32-CAM 拍照+JPEG 编码+WiFi 传输可能需要较长时间
        resp = requests.get(capture_url, stream=True, timeout=30)
        if resp.status_code == 200:
            # 使用 bytearray 分块收集响应数据
            # chunk_size=4096: 每次读取 4KB，平衡内存使用和传输效率
            image_data = bytearray()
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:  # 过滤掉心跳/空块
                    image_data.extend(chunk)

            if len(image_data) > 0:
                # 将字节数据转换为 numpy 数组（uint8 格式）
                # np.frombuffer 不复制数据，直接引用 bytearray 的内存（零拷贝）
                nparr = np.frombuffer(image_data, np.uint8)
                # cv2.imdecode 从内存中的 JPEG 数据解码为 BGR 图像
                # IMREAD_COLOR: 强制解码为三通道彩色图像（即使原图是灰度图）
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img, "camera"
    except Exception as e:
        # 捕获所有异常（网络超时、连接拒绝、DNS 解析失败等）
        print("获取图片失败: {}".format(e))

    # 所有获取方式均失败，返回空结果
    return None, "none"
