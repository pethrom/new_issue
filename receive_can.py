# -*- coding: utf-8 -*-
"""
CAN -> IoTDB 定时快照入库（10ms，所有字段永远保留最近值）
- 接收线程：解析 PGN，更新“最近值”缓存（统一大端 + 负偏移）
- 入库线程：每 10ms 对齐栅格写入；热身：等全部字段各出现一次或超时
- 运行阶段：不做“陈旧剔除”，每次入库都带齐所有字段，不写 None
"""

import time
import sys
import threading
from ctypes import *
from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType

# ================== 可调参数 ==================
PERIOD_MS = 10                  # 入库周期：10ms（100Hz）
IOTDB_HOST = "127.0.0.1"
IOTDB_PORT = "6667"
IOTDB_USER = "root"
IOTDB_PASS = "root"
IOTDB_PATH = "root.engine1"     # 单设备路径

# CAN 波特率（示例：1Mbps 的 SJA1000）
CAN_BAUD_TIMING0 = 0x00
CAN_BAUD_TIMING1 = 0x14

WARMUP_TIMEOUT_MS = 1200        # 全字段未齐 → 最多等 1.2s 就开始

# ================== ZLG USBCAN 定义 ==================
USBCAN_I = c_uint32(3)          # USBCAN-I/I+ 设备类型号

class ZCAN_CAN_OBJ(Structure):
    _fields_ = [
        ("ID", c_uint32),
        ("TimeStamp", c_uint32),
        ("TimeFlag", c_uint8),
        ("SendType", c_byte),
        ("RemoteFlag", c_byte),
        ("ExternFlag", c_byte),
        ("DataLen", c_byte),
        ("Data", c_ubyte * 8),
        ("Reserved", c_ubyte * 3),
    ]

class ZCAN_CAN_INIT_CONFIG(Structure):
    _fields_ = [
        ("AccCode", c_int), ("AccMask", c_int), ("Reserved", c_int),
        ("Filter", c_ubyte), ("Timing0", c_ubyte), ("Timing1", c_ubyte), ("Mode", c_ubyte)
    ]

def load_zlg_dll():
    """按平台加载 ZLG 动态库"""
    try:
        if sys.platform.startswith("win"):
            return cdll.LoadLibrary("ControlCAN.dll")
        else:
            return cdll.LoadLibrary("./libusbcan.so")
    except OSError as e:
        print("❌ 无法加载 ZLG 动态库：", e)
        print("   - Windows 请确认 ControlCAN.dll 在可搜索路径")
        print("   - Linux   请确认 ./libusbcan.so 在当前目录或 LD_LIBRARY_PATH")
        sys.exit(1)

dll = load_zlg_dll()

# ===== 声明函数签名 =====
from ctypes import POINTER
dll.VCI_GetReceiveNum.argtypes = [c_uint32, c_uint32, c_uint32]
dll.VCI_GetReceiveNum.restype  = c_uint32

dll.VCI_Receive.argtypes = [c_uint32, c_uint32, c_uint32, POINTER(ZCAN_CAN_OBJ), c_uint32, c_int]
dll.VCI_Receive.restype  = c_int

dll.VCI_OpenDevice.argtypes = [c_uint32, c_uint32, c_uint32]
dll.VCI_OpenDevice.restype  = c_uint32

dll.VCI_InitCAN.argtypes = [c_uint32, c_uint32, c_uint32, POINTER(ZCAN_CAN_INIT_CONFIG)]
dll.VCI_InitCAN.restype  = c_uint32

dll.VCI_StartCAN.argtypes = [c_uint32, c_uint32, c_uint32]
dll.VCI_StartCAN.restype  = c_uint32

dll.VCI_ClearBuffer.argtypes = [c_uint32, c_uint32, c_uint32]
dll.VCI_ClearBuffer.restype  = c_uint32

dll.VCI_CloseDevice.argtypes = [c_uint32, c_uint32]
dll.VCI_CloseDevice.restype  = c_uint32

# ================== IoTDB 会话与写入 ==================
def open_session():
    sess = Session(IOTDB_HOST, IOTDB_PORT, IOTDB_USER, IOTDB_PASS, fetch_size=1024)
    sess.open(False)
    return sess

def insert_row(session, timestamp_ms: int, measurements_order, values_map: dict):
    """
    固定列顺序写入；保证不写 None。
    measurements_order: 预定义的全字段顺序列表
    values_map: 字段 -> 值（必须已保证都有值）
    """
    measurements = list(measurements_order)
    vals = [float(values_map[m]) for m in measurements]  # 这里假定都存在且非 None
    types = [TSDataType.DOUBLE] * len(vals)
    session.insert_record(IOTDB_PATH, int(timestamp_ms), measurements, types, vals)

# ================== 大端 16bit 读取 ==================
def be_u16(d: bytes, i: int) -> int:
    return ((d[i] << 8) | d[i+1]) & 0xFFFF

# ================== 各 PGN 解码（统一大端 + 负偏移） ==================
def dec_18FEDF01(d: bytes):
    return {
        "AvgRPM":  float(be_u16(d, 0) * 1.0 - 0),
        "InstRPM": float(be_u16(d, 2) * 1.0 - 0),
    }

def dec_18FEDF02(d: bytes):
    return {
        "InAirP_kPa":   float(be_u16(d, 0) * 0.1 - 0),
        "ExhInP_kPa":   float(be_u16(d, 2) * 0.1 - 0),
        "OilP_bar":     float(be_u16(d, 4) * 0.1 - 0),
        "ExhOutP_kPa":  float(be_u16(d, 6) * 0.1 - 0),
    }

def dec_18FEDF03(d: bytes):
    return {
        "O2_percent":    float(be_u16(d, 0) * 0.000514 - 12),
        "NOx_ppm":       float(be_u16(d, 2) * 0.05     - 200),
        "InAirFlow_mA":  float(be_u16(d, 4) * 0.1      - 0),
        "FuelT_degC":    float(be_u16(d, 6) * 0.1      - 30),
    }

def dec_18FEDF04(d: bytes):
    return {
        "ExhInT_degC":   float(be_u16(d, 0) * 0.1 - 40),
        "ExhOutT_degC":  float(be_u16(d, 2) * 0.1 - 40),
        "InAirT_degC":   float(be_u16(d, 4) * 0.1 - 40),
        "OilT_degC":     float(be_u16(d, 6) * 0.1 - 0),
    }

def dec_18FEDF05(d: bytes):
    return {
        "EngColInT_degC":  float(be_u16(d, 0) * 0.1 - 30),
        "EngColOutT_degC": float(be_u16(d, 2) * 0.1 - 0),
        "EngT1_degC":      float(be_u16(d, 4) * 0.1 - 40),
        "EngT2_degC":      float(be_u16(d, 6) * 0.1 - 40),
    }

def dec_18FEDF07(d: bytes):
    return {
        "U_V":  float(be_u16(d, 0) * 0.1 - 0),
        "I_mA": float(be_u16(d, 2) * 0.1 - 50),
    }

DECODERS = {
    0x18FEDF01: dec_18FEDF01,
    0x18FEDF02: dec_18FEDF02,
    0x18FEDF03: dec_18FEDF03,
    0x18FEDF04: dec_18FEDF04,
    0x18FEDF05: dec_18FEDF05,
    0x18FEDF07: dec_18FEDF07,
}

# === 统一字段清单（固定写入顺序） ===
ALL_FIELDS = [
	# 18FEDF01
	"AvgRPM", "InstRPM",
    # 18FEDF02
    "InAirP_kPa", "ExhInP_kPa", "OilP_bar", "ExhOutP_kPa",
    # 18FEDF03
    "O2_percent", "NOx_ppm", "InAirFlow_mA", "FuelT_degC",
    # 18FEDF04
    "ExhInT_degC", "ExhOutT_degC", "InAirT_degC", "OilT_degC",
    # 18FEDF05
    "EngColInT_degC", "EngColOutT_degC", "EngT1_degC", "EngT2_degC",
    # 18FEDF07
    "U_V", "I_mA",
]

# 热身要求：等 ALL_FIELDS 全部出现一次（或超时）
WARMUP_REQUIRED_FIELDS = set(ALL_FIELDS)

# ================== 行缓存（永远保留最近值） ==================
class SnapshotCache:
    def __init__(self, fields_order):
        self.fields_order = list(fields_order)
        self.last_values = {}       # 字段 -> 最近工程值
        self.last_update = {}       # 字段 -> 最近更新时间戳(ms)
        self.lock = threading.Lock()

    def update_from_frame(self, can_id: int, data8: bytes):
        fn = DECODERS.get(can_id)
        if not fn:
            return
        vals = fn(data8)
        now_ms = int(time.time() * 1000)
        with self.lock:
            for k, v in vals.items():
                # 只要解码到值，就更新并覆盖为“最新值”
                self.last_values[k] = v
                self.last_update[k] = now_ms

    def warmup_ready(self, start_ms: int) -> bool:
        """热身是否就绪：ALL_FIELDS 是否都出现过；或超时"""
        with self.lock:
            has_all = all(k in self.last_values for k in WARMUP_REQUIRED_FIELDS)
        if has_all:
            return True
        return (int(time.time() * 1000) - start_ms) >= WARMUP_TIMEOUT_MS

    def build_full_row_no_null(self) -> dict:
        """
        返回一个“完整行”，包含 ALL_FIELDS 的每个字段。
        要求：这些字段在热身阶段已出现过一次，因此都应在 last_values 中。
        """
        with self.lock:
            # 如果有字段尚未出现（极端超时开始的情况），这里直接不返回行，避免写入空
            if not all(k in self.last_values for k in self.fields_order):
                return {}
            # 按固定顺序拷贝一份（确保后面 insert 时取到完整值）
            return {k: self.last_values[k] for k in self.fields_order}

# ================== 接收线程（仅更新缓存） ==================
def rx_thread(cache: SnapshotCache):
    print("策略：接收线程仅解析并更新缓存（统一大端 + 负偏移），不直接写库。")
    while True:
        try:
            cnt = dll.VCI_GetReceiveNum(USBCAN_I, 0, 0)
            if cnt > 0:
                arr = (ZCAN_CAN_OBJ * cnt)()
                rcv = dll.VCI_Receive(USBCAN_I, 0, 0, arr, cnt, 10)
                for i in range(max(0, rcv)):
                    c = arr[i]
                    # 可选：只要扩展帧
                    if c.ExternFlag != 1:
                        continue
                    n = max(0, min(int(c.DataLen), 8))
                    data8 = bytes(c.Data[:n])
                    if len(data8) < 8:
                        data8 += b"\x00" * (8 - len(data8))
                    cache.update_from_frame(c.ID, data8)
            else:
                time.sleep(0.002)  # 2ms 轮询
        except Exception as e:
            print("接收线程异常:", e)
            time.sleep(0.01)

# ================== 入库线程（固定 10ms 栅格 + 全字段写入） ==================
def flush_thread(session, cache: SnapshotCache):
    print(f"定时入库线程：每 {PERIOD_MS}ms 写入一次（时间戳对齐栅格，字段固定且无空）。")
    start_ms = int(time.time() * 1000)

    # 等待热身
    while not cache.warmup_ready(start_ms):
        time.sleep(0.005)

    base = int(time.time() * 1000)
    next_ts = ((base // PERIOD_MS) + 1) * PERIOD_MS

    while True:
        now = int(time.time() * 1000)
        sleep_ms = next_ts - now - 1
        if sleep_ms > 1:
            time.sleep(sleep_ms / 1000.0)
        while int(time.time() * 1000) < next_ts:
            time.sleep(0)

        row = cache.build_full_row_no_null()
        if row:
            try:
                insert_row(session, next_ts, cache.fields_order, row)

                # 🚀 打印关键字段
                print(
                    f"[{time.strftime('%H:%M:%S', time.localtime(next_ts/1000))}.{next_ts%1000:03d}] "
                    f"RPM={row['InstRPM']:.1f} | InAirP={row['InAirP_kPa']:.1f} kPa | "
                    f"InAirT={row['InAirT_degC']:.1f} °C | U={row['U_V']:.2f} V | I={row['I_mA']:.1f} mA"
                )

            except Exception as e:
                print("IoTDB 写入异常:", e)

        next_ts += PERIOD_MS
        late = int(time.time() * 1000) - next_ts
        if late > 5 * PERIOD_MS:
            now2 = int(time.time() * 1000)
            next_ts = ((now2 // PERIOD_MS) + 1) * PERIOD_MS

# ================== 主函数 ==================
def main():
    # 打开设备
    ret = dll.VCI_OpenDevice(USBCAN_I, 0, 0)
    if ret == 0:
        print("❌ 打开 USBCAN-I-mini 失败")
        return
    print("✅ 打开设备成功")

    cfg = ZCAN_CAN_INIT_CONFIG()
    cfg.AccCode = 0
    cfg.AccMask = 0xFFFFFFFF
    cfg.Filter  = 1          # 接收所有（需要可再改硬件过滤）
    cfg.Timing0 = CAN_BAUD_TIMING0
    cfg.Timing1 = CAN_BAUD_TIMING1
    cfg.Mode    = 0          # 正常模式

    dll.VCI_InitCAN(USBCAN_I, 0, 0, byref(cfg))
    dll.VCI_StartCAN(USBCAN_I, 0, 0)
    dll.VCI_ClearBuffer(USBCAN_I, 0, 0)

    # IoTDB 会话
    session = open_session()

    # 共享缓存（固定字段顺序 = ALL_FIELDS）
    cache = SnapshotCache(ALL_FIELDS)

    # 启动线程
    t_rx = threading.Thread(target=rx_thread, args=(cache,), daemon=True)
    t_rx.start()

    t_flush = threading.Thread(target=flush_thread, args=(session, cache), daemon=True)
    t_flush.start()

    print("接收线程 & ⏱️ 定时入库线程 已启动（10ms 栅格，无空值）。Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            session.close()
        except:
            pass
        try:
            dll.VCI_CloseDevice(USBCAN_I, 0)
        except:
            pass

if __name__ == "__main__":
    main()
