// =============================================================================
// 农作物病虫害监测系统 - ESP32-CAM 固件
// 功能：摄像头采集 + WiFi 图传 + SD 卡定时自动拍照存储
// =============================================================================

#include <Arduino.h>        // Arduino 核心库，提供 Serial、delay、millis 等基础函数
#include "esp_camera.h"     // ESP32 摄像头驱动库，提供摄像头初始化、帧捕获等 API
#include <WiFi.h>           // ESP32 WiFi 库，用于连接无线网络实现图传功能
#include <SD_MMC.h>         // SD 卡 MMC 模式驱动库，比 SPI 模式更快，用于本地存储照片
#include <time.h>           // C 标准时间库，用于 NTP 时间同步和生成带时间戳的文件名
// ===========================
// 摄像头型号选择（在 board_config.h 中定义引脚映射）
// board_config.h 包含不同 ESP32-CAM 板型的 GPIO 引脚定义
// ===========================
#include "board_config.h"

// ===========================
// WiFi 无线网络连接凭据
// ===========================
const char *ssid = "【请修改：你的WiFi名称】";      // WiFi 网络名称（SSID），部署时必须修改为实际路由器名称
const char *password = "【请修改：你的WiFi密码】"; // WiFi 连接密码，部署时必须修改为实际路由器密码

// ===========================
// 自动拍照存卡配置
// 用于定时采集田间图像，供后续病虫害分析使用
// ===========================
const unsigned long AUTO_SAVE_INTERVAL = 300000; // 自动拍照间隔，单位毫秒 (300000 = 5分钟拍一张存入SD卡)
unsigned long last_auto_save_time = 0;            // 上次自动拍照的时间戳，配合 millis() 实现非阻塞定时

// ===========================
// 函数声明（前向声明，实现在其他文件或本文件后部）
// ===========================
void startCameraServer();  // 启动摄像头 HTTP 流媒体服务器（定义在 app_httpd.cpp 中）
void setupLedFlash();      // 初始化板载 LED 闪光灯引脚（如果硬件支持）
void auto_save_photo();    // 自动拍照并保存到 SD 卡（带时间戳文件名）

void setup() {
  // ── 串口初始化 ──
  // 波特率 115200，用于调试输出和系统状态监控
  Serial.begin(115200);
  Serial.setDebugOutput(true);  // 开启调试输出，ESP-IDF 内部日志也会通过串口打印
  Serial.println();

  // ── 摄像头配置结构体 ──
  // camera_config_t 是 esp_camera 库定义的配置结构体，包含所有摄像头硬件参数
  camera_config_t config;
  // LEDC（LED 控制）通道和定时器：用于生成摄像头所需的 XCLK 时钟信号
  config.ledc_channel = LEDC_CHANNEL_0;  // 使用 LEDC 通道 0 输出 XCLK 时钟
  config.ledc_timer = LEDC_TIMER_0;      // 使用 LEDC 定时器 0
  // ── 数据线引脚映射（D0-D7，8位并行数据总线）──
  // 这些引脚定义在 board_config.h 中，不同板型引脚不同
  config.pin_d0 = Y2_GPIO_NUM;   // 数据位 D0（最低位）
  config.pin_d1 = Y3_GPIO_NUM;   // 数据位 D1
  config.pin_d2 = Y4_GPIO_NUM;   // 数据位 D2
  config.pin_d3 = Y5_GPIO_NUM;   // 数据位 D3
  config.pin_d4 = Y6_GPIO_NUM;   // 数据位 D4
  config.pin_d5 = Y7_GPIO_NUM;   // 数据位 D5
  config.pin_d6 = Y8_GPIO_NUM;   // 数据位 D6
  config.pin_d7 = Y9_GPIO_NUM;   // 数据位 D7（最高位）
  // ── 时钟和同步信号引脚 ──
  config.pin_xclk = XCLK_GPIO_NUM;    // 主时钟输出引脚，由 ESP32 提供给摄像头模组
  config.pin_pclk = PCLK_GPIO_NUM;    // 像素时钟引脚，摄像头模组输出，用于同步数据采样
  config.pin_vsync = VSYNC_GPIO_NUM;  // 垂直同步信号，标记一帧图像的开始
  config.pin_href = HREF_GPIO_NUM;    // 水平参考信号，标记一行有效像素数据
  // ── SCCB（I2C）控制总线引脚 ──
  // SCCB 是 OmniVision 摄像头的串行控制总线，用于配置摄像头寄存器
  config.pin_sccb_sda = SIOD_GPIO_NUM;  // SCCB 数据线（类似 I2C SDA）
  config.pin_sccb_scl = SIOC_GPIO_NUM;  // SCCB 时钟线（类似 I2C SCL）
  // ── 电源和复位控制引脚 ──
  config.pin_pwdn = PWDN_GPIO_NUM;    // 电源控制引脚（-1 表示该引脚不存在）
  config.pin_reset = RESET_GPIO_NUM;  // 硬件复位引脚（-1 表示使用软件复位）
  // ── 时钟频率 ──
  config.xclk_freq_hz = 20000000;  // XCLK 时钟频率 20MHz，是摄像头的数据采样时钟
                                    // 频率越高帧率越高但对信号质量要求也越高
  // ── 帧大小（分辨率）──
  config.frame_size = FRAMESIZE_UXGA; // UXGA = 1600x1200，初始化时使用最大分辨率
                                       // 后续会根据 PSRAM 情况和实际需要调整
  // ── 像素格式 ──
  config.pixel_format = PIXFORMAT_JPEG;  // JPEG 压缩格式，适合网络传输和文件存储
                                          // 硬件 JPEG 编码，ESP32 不消耗 CPU 做压缩
  //config.pixel_format = PIXFORMAT_RGB565; // RGB565 原始格式，适合人脸检测/识别（CPU直接处理）
  // ── 帧捕获模式 ──
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;  // 默认：仅在帧缓冲区为空时捕获新帧
                                               // 有 PSRAM 时会改为 CAMERA_GRAB_LATEST
  // ── 帧缓冲区存储位置 ──
  config.fb_location = CAMERA_FB_IN_PSRAM;  // 帧缓冲区放在 PSRAM（外部扩展 RAM）中
                                             // PSRAM 容量大，可存储高分辨率帧
  // ── JPEG 压缩质量 ──
  config.jpeg_quality = 10;  // JPEG 质量参数，范围 0-63，数值越小质量越高、文件越大
                              // 10 = 高质量，适合病虫害检测需要清晰细节的场景

  // ── 根据 PSRAM 存在与否调整配置 ──
  // PSRAM（伪静态随机存储器）是 ESP32-CAM 板载的外部扩展内存（通常 4MB）
  // 有 PSRAM 时可以使用更大的帧缓冲区和更高的分辨率
  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      // 检测到 PSRAM：可以使用双帧缓冲 + 最新帧捕获模式
      config.jpeg_quality = 10;  // 高质量，图片更清晰便于病虫害检测
      config.fb_count = 2;       // 双帧缓冲区，一个用于显示/传输，一个用于采集
      config.grab_mode = CAMERA_GRAB_LATEST;  // 始终获取最新帧，丢弃旧帧
                                               // 确保拍照时拿到的是最新画面
    } else {
      // 没有 PSRAM：内部 SRAM 有限（约 520KB），必须降低分辨率
      config.frame_size = FRAMESIZE_SVGA;          // 降为 SVGA (800x600) 以适应内存限制
      config.fb_location = CAMERA_FB_IN_DRAM;      // 帧缓冲区放在内部 DRAM 中
    }
  } else {
    // 非 JPEG 格式（RGB565）：用于人脸检测/识别场景
    // RGB565 每像素占 2 字节，帧数据量大，只能用很小的分辨率
    config.frame_size = FRAMESIZE_240X240;  // 240x240 小分辨率，适合实时处理
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;  // ESP32-S3 有足够的 PSRAM，可以双缓冲
#endif
  }

  // ── ESP-EYE 开发板特殊引脚配置 ──
  // ESP-EYE 是乐鑫官方的视觉开发板，GPIO13/14 连接了特殊功能按键
  // 配置为上拉输入，防止引脚悬空导致误触发
#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // ── 摄像头硬件初始化 ──
  // 调用 esp_camera 库的初始化函数，完成 SCCB 通信、传感器配置、帧缓冲区分配
  esp_err_t err = esp_camera_init(&config);
  Serial.printf("摄像头初始化: 0x%x\n", err);
  if (err != ESP_OK) {
    // 初始化失败：常见原因包括接线错误、摄像头模组损坏、PSRAM 故障等
    Serial.printf("摄像头初始化失败，错误码: 0x%x\n", err);
    return;  // 直接返回，不继续后续初始化（系统不可用）
  }
  // ── 测试帧捕获 ──
  // 获取一帧验证摄像头是否真正工作，并打印实际输出的分辨率
  // 有些传感器初始帧可能为空白，所以这里丢弃第一帧用于预热
  camera_fb_t *fb = esp_camera_fb_get();  // 从帧缓冲区获取一帧图像
  if (fb) {
    Serial.printf("实际分辨率: %d x %d\n", fb->width, fb->height);
    esp_camera_fb_return(fb);  // 归还帧缓冲区，释放给驱动继续使用
  }
  // ── PSRAM 内存信息打印 ──
  // 用于调试，确认 PSRAM 是否正确识别及剩余可用空间
  Serial.printf("PSRAM 大小: %d bytes\n", ESP.getPsramSize());   // PSRAM 总容量
  Serial.printf("可用 PSRAM: %d bytes\n", ESP.getFreePsram());   // 当前剩余可用容量

  sensor_t *s = esp_camera_sensor_get();  // 获取传感器控制句柄，用于后续调整摄像头参数
  // ── SD 卡初始化（放在 WiFi 连接之前，确保存储优先就绪）──
  // SD_MMC.begin 参数说明：
  //   "/sdcard" = 挂载点路径，后续所有 SD 卡文件操作使用此路径前缀
  //   true = 使用 1-bit 总线模式（只用 D0 数据线），比 4-bit 模式少占引脚
  //          ESP32-CAM 的 GPIO4 同时连接 SD 卡 D0 和板载 LED，1-bit 模式可避免冲突
  if (!SD_MMC.begin("/sdcard", true)) {
    Serial.println("SD 卡初始化失败!");
  } else {
    Serial.println("SD 卡初始化成功");
    // ── SD 卡类型检测 ──
    uint8_t cardType = SD_MMC.cardType();
    if (cardType == CARD_NONE) {
      Serial.println("未检测到 SD 卡");
    } else {
      // 打印 SD 卡类型：MMC = eMMC芯片, SDSC = 标准容量(<=2GB), SDHC = 高容量(4-32GB)
      Serial.print("SD 卡类型: ");
      if (cardType == CARD_MMC) Serial.println("MMC");
      else if (cardType == CARD_SD) Serial.println("SDSC");
      else if (cardType == CARD_SDHC) Serial.println("SDHC");
      else Serial.println("UNKNOWN");

      // ── SD 卡容量计算 ──
      // cardSize() 返回字节数，除以 1024*1024 转换为 MB
      uint64_t cardSize = SD_MMC.cardSize() / (1024 * 1024);
      Serial.printf("SD 卡容量: %llu MB\n", cardSize);
    }
  }
  // ── OV3660 传感器图像参数微调 ──
  // OV3660 是部分 ESP32-CAM 板型使用的摄像头模组（另有 OV2640 版本）
  // 默认输出图像存在垂直翻转和色彩过饱和问题，需要手动修正
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);        // 垂直翻转：将倒置的图像翻转回正常方向
    s->set_brightness(s, 1);   // 亮度微调：稍微提高亮度（范围 -2~2）
    s->set_saturation(s, -2);  // 饱和度降低：默认色彩过于浓艳，降低使颜色更真实
                                // 对病虫害检测很重要，需要准确的颜色信息
  }
  // ── 设置运行分辨率 ──
  // 初始化时用 UXGA 测试最大能力，实际运行时切换到 SVGA 平衡性能和画质
  if (config.pixel_format == PIXFORMAT_JPEG) {
    // SVGA (800x600) 是性价比最佳的选择：
    // - 足够清晰用于病虫害检测
    // - 帧率较高，实时预览流畅
    // - SD 卡单张图片约 50-100KB，存储压力小
    s->set_framesize(s, FRAMESIZE_SVGA);
  }

  // ── 创建 /data 存储目录 ──
  // 自动拍照的 JPEG 文件将保存在此目录下，文件名格式为 AUTO_YYYYMMDD_HHMMSS.jpg
  // 只在 SD 卡正常识别的情况下才尝试创建
  if (SD_MMC.cardType() != CARD_NONE) {
    if (!SD_MMC.exists("/data")) {       // 先检查目录是否已存在，避免重复创建
      if (SD_MMC.mkdir("/data")) {       // mkdir 只能创建一级目录（不支持递归创建）
        Serial.println("已创建 /data 目录");
      } else {
        Serial.println("创建 /data 目录失败!");  // 可能原因：SD 卡只读、空间已满
      }
    } else {
      Serial.println("/data 目录已存在");
    }
  }

  // ── M5Stack 系列开发板图像方向修正 ──
  // M5Stack 设备的摄像头安装方向不同，需要同时垂直翻转和水平镜像
#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);    // 垂直翻转
  s->set_hmirror(s, 1);  // 水平镜像
#endif

  // ── ESP32-S3-EYE 开发板图像方向修正 ──
#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);    // 垂直翻转
#endif

  // ── LED 闪光灯初始化 ──
  // 如果板型定义了 LED 引脚（LED_GPIO_NUM），则初始化闪光灯控制
  // ESP32-CAM 板载的 LED 同时复用了 SD 卡 D0 引脚（GPIO4），使用时需注意冲突
#if defined(LED_GPIO_NUM)
  setupLedFlash();
#endif

  // ── WiFi 无线连接 ──
  // 连接 WiFi 以实现图传功能（通过 HTTP 流在浏览器中实时查看摄像头画面）
  WiFi.begin(ssid, password);       // 启动 WiFi 连接（异步，不会立即连上）
  WiFi.setSleep(false);             // 禁用 WiFi 省电模式，保持连接稳定
                                     // 省电模式会导致连接间歇性断开，影响图传

  Serial.print("WiFi connecting");
  // 轮询等待 WiFi 连接成功，每 500ms 打印一个点表示进度
  // 注意：没有超时限制，如果密码错误会一直等待
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");  // WiFi 连接成功，可以获取 IP 地址了

  // ── NTP 网络时间同步 ──
  // 自动拍照的文件名需要精确的时间戳（年月日时分秒），必须从 NTP 服务器获取标准时间
  // 如果时间不准确，会导致文件名混乱，无法按时间顺序分析病虫害发展情况
  configTzTime("CST-8", "pool.ntp.org", "ntp.aliyun.com");
  // 参数说明：
  //   "CST-8"          = 时区设置，CST（中国标准时间）= UTC+8
  //   "pool.ntp.org"   = 主 NTP 服务器（国际通用 NTP 池）
  //   "ntp.aliyun.com" = 备用 NTP 服务器（阿里云提供，国内访问更快更稳定）

  Serial.println("正在同步 NTP 时间...");
  // 最多重试 20 次（共约 10 秒），等待时间同步完成
  int ntp_retry = 0;
  // 1700000000 = 2023-11-14 的时间戳，用于判断 NTP 是否返回了有效时间
  // ESP32 启动时 time(nullptr) 返回 0 或很小的值，表示时间未设置
  while (time(nullptr) < 1700000000 && ntp_retry < 20) {
    delay(500);
    Serial.print(".");
    ntp_retry++;
  }
  if (time(nullptr) >= 1700000000) {
    // NTP 同步成功，打印当前时间用于确认
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
      Serial.printf("\nNTP 同步成功: %s", asctime(&timeinfo));
      // asctime 输出格式如 "Tue Nov 14 22:13:04 2023\n"
    }
  } else {
    // NTP 同步失败：可能原因包括网络不通、DNS 解析失败、NTP 服务器无响应
    // 此时拍照仍会进行，但文件名时间戳将不准确（从 1970 年开始计时）
    Serial.println("\nNTP 同步失败，将使用默认时间");
  }

  // ── 启动摄像头 HTTP 流媒体服务器 ──
  // 在 ESP32 上创建一个 HTTP 服务器，提供以下功能：
  // - 视频流页面：浏览器可实时查看摄像头画面（MJPEG 流）
  // - 控制接口：可远程调整分辨率、亮度、对比度等参数
  // 服务器实现定义在 app_httpd.cpp 文件中
  startCameraServer();

  // 打印访问地址：用户在浏览器输入此 IP 即可打开摄像头控制页面
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());  // 获取并打印 ESP32 的局域网 IP 地址
  Serial.println("' to connect");
}

// =============================================================================
// 主循环函数 - 系统启动后反复执行
// =============================================================================
void loop() {
  // ── 非阻塞定时器机制 ──
  // 使用 millis()（系统启动以来的毫秒数）代替 delay() 实现定时
  // 优点：不会阻塞 CPU，摄像头服务器可以持续在后台运行
  unsigned long now = millis();  // 获取当前系统运行时间（毫秒）

  // 判断是否到达自动拍照时间点
  // 使用差值比较而非直接比较，正确处理 millis() 溢出（约 49 天后回绕到 0）
  if (now - last_auto_save_time >= AUTO_SAVE_INTERVAL) {
    auto_save_photo();           // 执行一次自动拍照存卡
    last_auto_save_time = now;   // 更新上次拍照时间，开始下一个计时周期
  }

  delay(1000);  // 每秒检查一次（1000ms），降低 CPU 占用率
                 // 精度足够：5 分钟间隔不需要毫秒级精度
}

// =============================================================================
// 自动拍照并保存到 SD 卡
// 由 loop() 中的定时器每 AUTO_SAVE_INTERVAL（5分钟）调用一次
// 文件名格式: /data/AUTO_YYYYMMDD_HHMMSS.jpg
// =============================================================================
void auto_save_photo() {
  // ── 第一步：检查 SD 卡是否就绪 ──
  // 如果没有插入 SD 卡或 SD 卡初始化失败，直接跳过，不拍照
  if (SD_MMC.cardType() == CARD_NONE) {
    Serial.println("[自动拍照] SD卡未就绪，跳过");
    return;
  }

  // ── 第二步：检查 NTP 时间是否已同步 ──
  // 确保文件名中的时间戳准确（>= 1700000000 即 2023-11-14 之后才算有效）
  // 如果时间未同步，文件名会变成 1970 年，导致后续按时间检索困难
  if (time(nullptr) < 1700000000) {
    Serial.println("[自动拍照] 时间未同步，跳过");
    return;
  }

  Serial.println("[自动拍照] 开始拍照存卡...");

  // ── 第三步：从摄像头获取一帧图像 ──
  // esp_camera_fb_get() 返回帧缓冲区指针，包含图像数据指针和长度
  // 如果 PSRAM 配置了 GRAB_LATEST 模式，这里获取的是最新拍摄的画面
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[自动拍照] 获取帧失败");  // 可能是摄像头驱动异常
    return;
  }

  // ── 第四步：获取当前时间并构建文件名 ──
  struct tm timeinfo;
  getLocalTime(&timeinfo);  // 获取本地时间（已按 CST-8 时区转换）

  // 使用 strftime 格式化时间为文件名
  // 示例输出: /data/AUTO_20241201_143025.jpg
  // 命名规则: AUTO_ 前缀 + 年月日 + 下划线 + 时分秒 + .jpg 扩展名
  char path[64];
  strftime(path, sizeof(path), "/data/AUTO_%Y%m%d_%H%M%S.jpg", &timeinfo);

  // ── 第五步：打开 SD 卡文件并写入图像数据 ──
  File file = SD_MMC.open(path, FILE_WRITE);  // 以写入模式打开文件（如不存在则创建）
  if (!file) {
    Serial.printf("[自动拍照] 打开文件失败: %s\n", path);  // 可能原因：目录不存在、磁盘满
    esp_camera_fb_return(fb);  // 归还帧缓冲区（重要！不归还会导致内存泄漏）
    return;
  }

  // 将帧缓冲区中的 JPEG 数据一次性写入文件
  // fb->buf = 图像数据指针（JPEG 编码后的字节流）
  // fb->len = 图像数据长度（字节数）
  size_t written = file.write(fb->buf, fb->len);  // 实际写入的字节数
  size_t fb_len = fb->len;  // 保存原始帧长度，用于后续校验
  file.close();              // 关闭文件，确保数据刷入 SD 卡
  esp_camera_fb_return(fb);  // 归还帧缓冲区给摄像头驱动

  // ── 第六步：验证写入完整性 ──
  // 比较实际写入字节数与帧数据长度，不一致说明写入中断（如 SD 卡突然拔出）
  if (written != fb_len) {
    Serial.printf("[自动拍照] 写入不完整! 需要写 %d, 实际写 %d，删除损坏文件\n", fb_len, written);
    SD_MMC.remove(path);  // 删除损坏的文件，避免后续处理时读取到不完整数据
  } else {
    Serial.printf("[自动拍照] 成功! 文件: %s, 大小: %d 字节\n", path, written);
  }
}
