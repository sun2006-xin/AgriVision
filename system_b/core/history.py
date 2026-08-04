"""
检测历史记录管理模块

负责记录每次虫害检测的结果数据（等级、病斑数、虫害数、占比等），
并提供查询、统计、趋势分析和数据导出功能。
所有记录以 JSON 格式持久化存储，关联的标注图像保存在独立的 images 子目录中。
"""

import json
import os
import threading
import zipfile
import shutil
from datetime import datetime, timedelta
from collections import defaultdict

# 历史记录的根目录，位于本模块所在目录下的 detection_logs/ 文件夹
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_logs")
# 历史记录 JSON 文件路径：detection_logs/history.json
# 存储所有检测记录的数组，每条记录包含时间戳、等级、病斑/虫害统计等
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
# 标注图像的存放目录：detection_logs/images/
# 仅当检测结果等级 >= "注意" 时才会保存对应的标注图像，正常结果不保存图像以节省空间
IMAGE_DIR = os.path.join(HISTORY_DIR, "images")


class HistoryManager:
    """
    检测历史记录管理器

    核心职责：
    - 每次检测完成后，将结果（等级、病斑数、虫害数、各项占比等）作为一条记录写入历史
    - 等级达到"注意"及以上时，额外保存标注后的图像以供后续复查
    - 提供按日期/等级筛选的分页查询、7天趋势统计、汇总统计、ZIP 数据导出等功能
    - 自动清理 30 天前的旧记录及其关联图像，防止磁盘空间无限增长

    数据存储在 detection_logs/ 目录下：
    - history.json       — 所有检测记录的 JSON 数组
    - images/            — 标注图像文件（文件名格式：YYYYMMDD_HHMMSS.jpg）

    典型用法：
        history_mgr = HistoryManager()
        record_id = history_mgr.add_record(result, annotated_image)
        page = history_mgr.query_records(date_from="2025-01-01", page=1)
        trend = history_mgr.get_trend_data(days=7)
    """

    def __init__(self):
        """
        初始化历史记录管理器

        初始化顺序：
        1. _ensure_dirs()  — 先确保 detection_logs/ 和 detection_logs/images/ 目录存在
        2. _load_history() — 再从 JSON 文件加载已有的历史记录到内存
        """
        self._lock = threading.Lock()  # 保护 records 列表和 history.json 的并发读写
        self._id_counter = 0           # 同一秒内多条记录的序列号，保证 ID 唯一
        self._last_id_second = ""      # 上一次生成 ID 的秒级时间戳
        self._ensure_dirs()
        self._load_history()
    
    def _ensure_dirs(self):
        """
        确保历史记录目录和图像子目录都已存在

        同时创建 HISTORY_DIR（检测日志根目录）和 IMAGE_DIR（图像子目录），
        exist_ok=True 保证目录已存在时不报错。
        """
        os.makedirs(HISTORY_DIR, exist_ok=True)
        os.makedirs(IMAGE_DIR, exist_ok=True)

    def _load_history(self):
        """
        从 JSON 文件加载历史记录到内存

        加载策略：
        1. 文件存在且格式正确 → 加载到 self.records 列表
        2. 文件存在但 JSON 损坏或 I/O 失败 → 静默回退为空列表（不阻塞程序启动）
        3. 文件不存在（首次运行） → 初始化为空列表

        注意：与 ConfigManager 不同，历史记录加载失败时仅打印警告而不抛异常，
        因为历史数据丢失不应阻止检测功能正常运行。
        """
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                # JSON 损坏或读取失败，丢弃旧数据从头开始
                self.records = []
        else:
            self.records = []

    def _save_history(self):
        """
        将内存中的全部记录持久化到 JSON 文件

        使用 ensure_ascii=False 保证中文字段（如等级名称"正常"/"注意"等）正常显示，
        indent=2 使文件便于人工查看和调试。
        写入失败时仅打印错误信息，不抛出异常（避免中断检测主流程）。
        """
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"保存历史记录失败: {e}")
    
    def add_record(self, result, image=None, dual_result=None):
        """
        添加一条新的检测记录

        :param result: 检测结果字典（由检测引擎返回，包含 level、disease_count 等字段）
        :param image: 标注后的 OpenCV 图像（numpy 数组），可选
        :param dual_result: 双引擎融合结果字典（由 DualVerifier.verify 返回），可选
        :return: 新记录的唯一 ID（record_id）
        """
        with self._lock:
            return self._add_record_unlocked(result, image, dual_result)

    def _add_record_unlocked(self, result, image=None, dual_result=None):
        """内部方法: 在已持有 _lock 时执行记录添加（调用方必须已持有 self._lock）"""
        # 生成唯一 ID: 同一秒内多条记录自动追加序列号
        now = datetime.now()
        second_str = now.strftime("%Y%m%d_%H%M%S")
        if second_str == self._last_id_second:
            self._id_counter += 1
        else:
            self._id_counter = 0
            self._last_id_second = second_str
        record_id = f"{second_str}_{self._id_counter}" if self._id_counter > 0 else second_str

        # 构造完整的记录字典，从 result 中提取各项检测数据，缺失字段使用安全默认值
        record = {
            "id": record_id,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": result.get("level", "正常"),
            "level_code": result.get("level_code", 0),
            "disease_count": result.get("disease_count", 0),
            "disease_ratio": result.get("disease_ratio", 0.0),
            "white_count": result.get("white_count", 0),
            "white_ratio": result.get("white_ratio", 0.0),
            "green_ratio": result.get("green_ratio", 0.0),
            "image_path": None,
            # 双引擎字段（dual_result 为 None 时写入空值，保持字段结构一致）
            "yolo_disease_count": 0,
            "yolo_bug_count": 0,
            "dual_confidence": "",
            "dual_agree_disease": "",
            "dual_agree_bug": "",
        }

        # 填充双引擎数据
        if dual_result is not None:
            yr = dual_result.get('yolo_result', {})
            record['yolo_disease_count'] = yr.get('disease_count', 0)
            record['yolo_bug_count'] = yr.get('bug_count', 0)
            record['dual_confidence'] = dual_result.get('confidence', '')
            agreement = dual_result.get('agreement', {})
            record['dual_agree_disease'] = agreement.get('disease', '')
            record['dual_agree_bug'] = agreement.get('bug', '')

        # 条件保存图像：仅当等级 >= "注意"（level_code >= 1）且确实传入了图像时
        level_code = result.get("level_code", 0)
        if level_code >= 1 and image is not None:
            image_filename = f"{record_id}.jpg"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            try:
                import cv2
                cv2.imwrite(image_path, image)  # 将标注后的图像保存为 JPG
                record["image_path"] = image_path
            except Exception as e:
                print(f"保存图像失败: {e}")

        # 将新记录插入列表头部，保证最新记录始终在最前面
        self.records.insert(0, record)

        # 自动清理超过 30 天的旧记录及其图像文件，防止 history.json 无限膨胀
        self._cleanup_old_records(days=30)

        # 持久化到磁盘
        self._save_history()

        return record_id
    
    def _cleanup_old_records(self, days=30):
        """
        清理超过指定天数的旧记录及其关联图像文件

        保留策略：默认保留最近 30 天的记录。
        - 日期比较基于字符串（"YYYY-MM-DD" 格式的字符串可直接用 < 比较大小）
        - 先按截止日期将记录分为"旧记录"和"新记录"两组
        - 遍历旧记录，逐一删除其关联的图像文件（如果存在）
        - 最后只保留新记录，旧的 JSON 条目被丢弃

        调用时机：每次 add_record() 插入新记录后自动调用。

        :param days: 保留天数，默认 30 天
        """
        # 计算截止日期：30 天前的那一天，格式 "YYYY-MM-DD"
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # 按日期字符串比较，将记录分为过期和有效两组
        old_records = [r for r in self.records if r["date"] < cutoff_date]
        new_records = [r for r in self.records if r["date"] >= cutoff_date]

        # 遍历过期记录，删除其关联的图像文件以释放磁盘空间
        for record in old_records:
            if record.get("image_path") and os.path.exists(record["image_path"]):
                try:
                    os.remove(record["image_path"])  # 删除旧图像
                except Exception as e:
                    print(f"删除旧图像失败: {e}")

        # 仅保留有效期内的记录
        self.records = new_records
    
    def query_records(self, date_from=None, date_to=None, level=None, page=1, page_size=20):
        """
        按条件查询历史记录（支持日期范围、等级筛选和分页）

        查询流程：
        1. 日期范围筛选 — 保留 date_from <= 记录日期 <= date_to 的记录
        2. 等级筛选 — 在日期筛选结果上进一步按等级名称过滤
        3. 分页 — 将筛选后的结果按 page_size 切片，返回指定页的数据

        日期比较基于 "YYYY-MM-DD" 字符串的字典序，无需转换为 datetime 对象。
        由于 self.records 本身按时间倒序排列（新记录在前），查询结果也保持此顺序。

        :param date_from: 起始日期字符串（含），如 "2025-01-01"，为 None 时不限起始
        :param date_to: 结束日期字符串（含），如 "2025-06-30"，为 None 时不限结束
        :param level: 等级筛选，可选值为 "正常"/"注意"/"警告"/"严重"，为 None 时不过滤
        :param page: 页码，从 1 开始
        :param page_size: 每页记录数，默认 20 条
        :return: 包含以下字段的字典：
                 - records      : 当前页的记录列表
                 - total        : 筛选后的总记录数
                 - pages        : 总页数
                 - current_page : 当前页码
                 - page_size    : 每页大小
        """
        with self._lock:
            filtered = list(self.records)

        # 日期范围筛选：利用 "YYYY-MM-DD" 字符串可直接比较大小
        if date_from:
            filtered = [r for r in filtered if r["date"] >= date_from]
        if date_to:
            filtered = [r for r in filtered if r["date"] <= date_to]

        # 等级筛选：精确匹配等级名称
        if level is not None:
            filtered = [r for r in filtered if r["level"] == level]

        # 计算分页参数
        total = len(filtered)
        pages = (total + page_size - 1) // page_size  # 向上取整计算总页数

        # 按页码切片，取出当前页的数据
        start = (page - 1) * page_size
        end = start + page_size
        paginated = filtered[start:end]
        
        return {
            "records": paginated,
            "total": total,
            "pages": pages,
            "current_page": page,
            "page_size": page_size
        }
    
    def get_trend_data(self, days=7):
        """
        获取最近 N 天的趋势数据（供前端 Chart.js 图表使用）

        遍历逻辑：
        - 从今天往前推 days 天，按时间正序（从旧到新）逐日统计
        - 对每一天：汇总病斑总数、虫害总数、以及各等级的记录计数
        - 没有记录的日期，对应数值为 0（保证数组长度与日期数组一致）

        返回的数据结构直接对应 Chart.js 的 datasets：
        - dates           → x 轴标签数组
        - disease_counts  → 病斑数量折线图的数据数组
        - pest_counts     → 虫害数量折线图的数据数组
        - level_stats     → 各等级柱状图的分组数据（每天一个 {正常:n, 注意:n, 警告:n, 严重:n} 字典）

        :param days: 统计天数，默认 7 天（一周趋势）
        :return: 包含 dates、disease_counts、pest_counts、level_stats 四个数组的字典
        """
        with self._lock:
            records_snapshot = list(self.records)
        end_date = datetime.now()
        dates = []           # x 轴日期标签，从旧到新排列
        disease_counts = []  # 每天的病斑总数
        pest_counts = []     # 每天的虫害总数
        level_stats = []     # 每天的各等级记录计数

        # 从第 days-1 天前开始到今天（倒序遍历 i，生成正序日期）
        for i in range(days - 1, -1, -1):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            dates.append(date)

            # 筛选出当天的所有记录
            day_records = [r for r in records_snapshot if r["date"] == date]

            # 汇总当天的病斑总数和虫害总数
            total_disease = sum(r["disease_count"] for r in day_records)
            total_pest = sum(r["white_count"] for r in day_records)
            disease_counts.append(total_disease)
            pest_counts.append(total_pest)

            # 统计当天各等级的记录数（初始化四个等级为 0，再逐个累加）
            level_counts = {"正常": 0, "注意": 0, "警告": 0, "严重": 0}
            for r in day_records:
                lvl = r.get("level", "正常")
                if lvl in level_counts:
                    level_counts[lvl] += 1
            level_stats.append(level_counts)
        
        return {
            "dates": dates,
            "disease_counts": disease_counts,
            "pest_counts": pest_counts,
            "level_stats": level_stats
        }
    
    def get_statistics(self, date_from=None, date_to=None):
        """
        获取指定日期范围内的汇总统计数据

        统计内容：
        1. total_records    — 该范围内的检测总次数
        2. level_counts     — 各等级（正常/注意/警告/严重）的检测次数分布
        3. total_disease    — 累计病斑检测总数
        4. total_pest       — 累计虫害检测总数
        5. avg_disease_ratio — 平均病斑面积占比（反映病害严重程度）
        6. avg_pest_ratio   — 平均虫害面积占比
        7. avg_green_ratio  — 平均绿色区域占比（反映整体健康水平）

        平均值计算时，若记录数为 0 则返回 0.0，避免除零错误。

        :param date_from: 起始日期（含），为 None 时不限起始
        :param date_to: 结束日期（含），为 None 时不限结束
        :return: 统计摘要字典
        """
        with self._lock:
            filtered = list(self.records)

        # 按日期范围筛选记录
        if date_from:
            filtered = [r for r in filtered if r["date"] >= date_from]
        if date_to:
            filtered = [r for r in filtered if r["date"] <= date_to]

        total_records = len(filtered)

        # 统计各等级的记录数
        level_counts = {"正常": 0, "注意": 0, "警告": 0, "严重": 0}
        for r in filtered:
            lvl = r.get("level", "正常")
            if lvl in level_counts:
                level_counts[lvl] += 1

        # 累计病斑和虫害检测总数
        total_disease = sum(r["disease_count"] for r in filtered)
        total_pest = sum(r["white_count"] for r in filtered)

        # 计算各项平均占比，记录数为 0 时安全返回 0.0
        avg_disease_ratio = sum(r["disease_ratio"] for r in filtered) / total_records if total_records > 0 else 0.0
        avg_pest_ratio = sum(r["white_ratio"] for r in filtered) / total_records if total_records > 0 else 0.0
        avg_green_ratio = sum(r["green_ratio"] for r in filtered) / total_records if total_records > 0 else 0.0
        
        return {
            "total_records": total_records,
            "level_counts": level_counts,
            "total_disease": total_disease,
            "total_pest": total_pest,
            "avg_disease_ratio": avg_disease_ratio,
            "avg_pest_ratio": avg_pest_ratio,
            "avg_green_ratio": avg_green_ratio
        }
    
    def export_data(self, date_from=None, date_to=None):
        """
        将指定日期范围内的检测数据导出为 ZIP 压缩包

        ZIP 包内容：
        1. detection_data.json — 筛选后的全部检测记录（JSON 数组）
        2. statistics.json     — 同一范围的汇总统计数据
        3. images/             — 关联的标注图像文件副本

        导出流程：
        1. 按日期范围筛选记录
        2. 在 detection_logs/export_temp/ 下创建临时目录
        3. 写入检测数据和统计摘要的 JSON 文件
        4. 将关联图像文件复制到临时目录的 images/ 子目录
        5. 将临时目录打包为 ZIP（使用 ZIP_DEFLATED 压缩）
        6. 在 finally 块中清理临时目录，确保不留垃圾文件

        使用 try/finally 保证无论打包是否成功，临时目录都会被清理。

        :param date_from: 起始日期（含），为 None 时不限起始
        :param date_to: 结束日期（含），为 None 时不限结束
        :return: 生成的 ZIP 文件路径（如 "detection_logs/detection_export_20250615_143022.zip"）
        """
        # 按日期范围筛选待导出的记录
        with self._lock:
            filtered = list(self.records)

        if date_from:
            filtered = [r for r in filtered if r["date"] >= date_from]
        if date_to:
            filtered = [r for r in filtered if r["date"] <= date_to]

        # 创建临时导出目录，用于暂存待打包的文件
        export_dir = os.path.join(HISTORY_DIR, "export_temp")
        os.makedirs(export_dir, exist_ok=True)

        try:
            # 步骤1：导出检测记录数据为 JSON
            data_file = os.path.join(export_dir, "detection_data.json")
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)

            # 步骤2：导出同一范围的统计摘要
            stats = self.get_statistics(date_from, date_to)
            stats_file = os.path.join(export_dir, "statistics.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            # 步骤3：复制关联的图像文件到临时目录
            image_export_dir = os.path.join(export_dir, "images")
            os.makedirs(image_export_dir, exist_ok=True)

            for record in filtered:
                if record.get("image_path") and os.path.exists(record["image_path"]):
                    try:
                        shutil.copy2(record["image_path"], image_export_dir)  # copy2 保留文件元数据
                    except Exception as e:
                        print(f"复制图像失败: {e}")

            # 步骤4：将临时目录中的所有文件打包为 ZIP 压缩包
            zip_filename = f"detection_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = os.path.join(HISTORY_DIR, zip_filename)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 遍历临时目录，将每个文件添加到 ZIP 中
                for root, dirs, files in os.walk(export_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 使用相对路径作为 ZIP 内的归档名，避免包含临时目录路径
                        arcname = os.path.relpath(file_path, export_dir)
                        zf.write(file_path, arcname)

            return zip_path

        finally:
            # 无论成功与否，都清理临时目录，防止磁盘空间泄漏
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)
    
    def get_record_by_id(self, record_id):
        """
        根据记录 ID 查找单条检测记录

        通过线性遍历 self.records 列表进行匹配。
        由于记录总量受 30 天保留策略限制（通常不超过数千条），线性查找的性能完全可接受。

        常用于 Web 前端点击某条记录后查看详情的场景。

        :param record_id: 记录唯一标识（格式为 YYYYMMDD_HHMMSS 的字符串）
        :return: 匹配的记录字典，未找到时返回 None
        """
        with self._lock:
            for record in self.records:
                if record["id"] == record_id:
                    return record
            return None

    def get_all_levels(self):
        """
        获取系统支持的所有检测等级列表

        返回固定的四个等级名称，常用于：
        - Web 前端等级筛选下拉框的选项数据源
        - 统计图表中等级维度的标签

        :return: 等级名称列表 ["正常", "注意", "警告", "严重"]
        """
        return ["正常", "注意", "警告", "严重"]

    def get_total_count(self):
        """
        获取当前内存中的总记录数

        直接返回 self.records 列表长度，无需遍历或计算。
        常用于前端显示"共 N 条记录"或在仪表板上展示记录总量。

        :return: 记录总数（整数）
        """
        with self._lock:
            return len(self.records)
