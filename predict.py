# -*- coding: utf-8 -*-
"""
IoTDB 增量拉取 + 忽略最新5行 + 行完整 + 严格10ms + 非重叠6帧窗口推理 + Tk UI
带四类调试输出 + 缺帧前向补帧(Forward Fill)：
1) 行不完整：    [DEBUG][IncompleteRow]
2) 连续性等待：  [DEBUG][GapWait]
3) 水印/候选：   [DEBUG][Watermark]
4) 队列满：      [DEBUG][QueueFull]
5) 补帧日志：    [FILL][GapFF]
"""

import os
import time
import csv
import queue
import threading
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import tkinter as tk
from iotdb.Session import Session

# === 你项目内模块 ===
from model import SimpleCNN_LSTM
from ui_realtime import RealtimeUI

# ================= 配置 =================
MODEL_PATH = "model/cnn-lstm.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPECT_DT_MS = 10             # 期望时间步长（严格10ms）
TAIL_IGNORE_ROWS = 5          # 每批忽略最新5行（≈50ms）

WINDOW_LEN = 6                # 窗口长度（与训练一致）
QUERY_LIMIT = 1024
POLL_INTERVAL_SEC = 0.3
RETRY_WAIT_SEC = 2.0

# --- 缺帧处理（前向补帧） ---
GAP_MAX_WAIT_MS = 500         # 等待缺帧的最大时长；超过则开始补
GAP_FILL_MAX_STEPS = 24       # 一次最多补多少帧（24*10ms=240ms）

# IoTDB
IOTDB_HOST = "127.0.0.1"
IOTDB_PORT = "6667"
IOTDB_USER = "root"
IOTDB_PASS = "root"
IOTDB_DB   = "root.engine1"

# 列映射
COL_WATER   = "EngT1_degC"   # 水温
COL_INAIR_T = "InAirT_degC"  # 进气温度
COL_EXH_T   = "ExhOutT_degC" # 排气温度
COL_INAIR_P = "InAirP_kPa"   # 进气压力
COL_O2      = "O2_percent"   # O2%
IOTDB_SELECT_COLUMNS = [COL_WATER, COL_INAIR_T, COL_EXH_T, COL_INAIR_P, COL_O2]

# ================= 归一化器（优先加载训练时保存的scaler） =================
FEATURE_SCALER_PATH = "model/feature_scaler.pkl"
TARGET_SCALER_PATH  = "model/target_scaler.pkl"

_loaded_feature_scaler = None
_loaded_target_scaler  = None
try:
    import joblib
    if os.path.exists(FEATURE_SCALER_PATH):
        _loaded_feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    if os.path.exists(TARGET_SCALER_PATH):
        _loaded_target_scaler  = joblib.load(TARGET_SCALER_PATH)
except Exception:
    _loaded_feature_scaler = None
    _loaded_target_scaler  = None

# 回退 min/max（若未提供scaler）
feature_min = np.array([-0.002097, 48.2, 43.4, 71.5, 19.1], dtype=np.float32)
feature_max = np.array([ 1.1467304, 81.5, 469.6,109.4, 24.5], dtype=np.float32)
EFF_MIN, EFF_MAX = 0.0, 29.23215

def normalize_feat(x: np.ndarray) -> np.ndarray:
    if _loaded_feature_scaler is not None:
        mn = _loaded_feature_scaler.data_min_
        mx = _loaded_feature_scaler.data_max_
        d = (mx - mn); d[d == 0] = 1.0
        return (x - mn) / d
    d = (feature_max - feature_min); d[d == 0] = 1.0
    return (x - feature_min) / d

def denormalize_eff(y_norm: np.ndarray) -> np.ndarray:
    if _loaded_target_scaler is not None:
        mn = _loaded_target_scaler.data_min_[0]
        mx = _loaded_target_scaler.data_max_[0]
        return y_norm * (mx - mn) + mn
    return y_norm * (EFF_MAX - EFF_MIN) + EFF_MIN

# ================= 物理量计算 =================
def o2_to_oil_flow(o2_percent: float) -> float:
    """O2% → 油量(g/s)"""
    Q2 = float(o2_percent)
    denom = 0.2095 - Q2 * 0.01
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6
    U2 = (0.2098 + 0.2737 * Q2 * 0.01) / denom
    if U2 <= 1e-6:
        U2 = 1e-6
    oil = 17.0 / (U2 * 14.8)
    return max(float(oil), 0.0)

def calc_power_w(oil_flow_gps: float, efficiency_percent: float) -> float:
    return float(oil_flow_gps) * 42600.0 * (float(efficiency_percent) / 100.0)

# ================= 队列 / 控制 / CSV =================
# 大缓存：存“完成且严格10ms连续/或补帧后的”单行
cache_queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(maxsize=50000)
ui_queue:    "queue.Queue[dict]" = queue.Queue(maxsize=5000)

RUNNING = threading.Event()  # 开始/暂停
RUNNING.clear()

# 计数器（真实吞吐监控）
put_count = 0
get_count = 0
pc_lock = threading.Lock()
gc_lock = threading.Lock()

# CSV
BATCH_CSV_PATH = "online_predict_results.csv"
BATCH_FLUSH_N = 200
_batch_cache = []

SINGLE_CSV_PATH = "predict_log.csv"
_single_csv_file = None
_single_csv_writer = None

def _ensure_single_csv():
    global _single_csv_file, _single_csv_writer
    if _single_csv_file is None:
        need_header = not os.path.exists(SINGLE_CSV_PATH)
        _single_csv_file = open(SINGLE_CSV_PATH, "a", newline="", encoding="utf-8")
        _single_csv_writer = csv.writer(_single_csv_file)
        if need_header:
            _single_csv_writer.writerow([
                "Time", "Prediction(%)", "Power(W)",
                "water_temp", "in_air_temp", "exh_temp",
                "in_air_pressure", "oil_flow(g/s)"
            ])

def _write_single_csv(item: dict):
    _ensure_single_csv()
    _single_csv_writer.writerow([
        item["Time"],
        f"{item['Prediction']:.2f}",
        f"{item['Power']:.2f}",
        f"{item['water_temp']:.2f}",
        f"{item['in_air_temp']:.2f}",
        f"{item['exh_temp']:.2f}",
        f"{item['in_air_pressure']:.2f}",
        f"{item['oil_flow']:.2f}",
    ])
    _single_csv_file.flush()

def _write_batch_csv_flush():
    if not _batch_cache:
        return
    df = pd.DataFrame(_batch_cache, columns=["Time", "Prediction"])
    header = not os.path.exists(BATCH_CSV_PATH)
    df.to_csv(BATCH_CSV_PATH, mode="a", header=header, index=False, encoding="utf-8-sig")
    _batch_cache.clear()

# ================= 采集线程（含四类DEBUG + 前向补帧） =================
def iotdb_fetch_thread():
    """
    - 按“开始”对齐 baseline_ts（当前最新行）
    - 增量：只查 Time > baseline_ts
    - 每批忽略最新5行（≈50ms，不稳定区）
    - 行完整才发；严格10ms连续；若缺帧超过阈值 => 用上一帧前向补齐
    - 发到 cache_queue（阻塞put）
    - Debug：IncompleteRow / GapWait / Watermark / QueueFull / FILL
    """
    print("采集线程启动：baseline对齐 + 忽略最新5行 + 行完整 + 严格10ms + 缺帧前向补帧 + 大缓存队列")

    sel = ", ".join(IOTDB_SELECT_COLUMNS)
    session = None
    staging: Dict[int, Dict[str, float]] = {}  # ts -> {col: value}

    baseline_ts = -1
    last_emitted_ts = None
    next_candidate_ts = None

    # 缺帧等待状态
    gap_since_ms = None         # 当前这次缺帧从何时开始等待
    gap_expected_ts = None      # 正在等待的期望 ts
    last_real_feats = None      # 最近一次“真实发射”的特征（用于前向填充）

    need_cols = [COL_WATER, COL_INAIR_T, COL_EXH_T, COL_INAIR_P, COL_O2]

    def getv(row, meas):
        full = f"{IOTDB_DB}.{meas}"
        return row[full] if full in row else row.get(meas, None)

    def probe_latest_ts(sess) -> int:
        _sql = f"SELECT {sel} FROM {IOTDB_DB} ORDER BY Time DESC LIMIT 1"
        _ds = sess.execute_query_statement(_sql)
        _df = _ds.todf()
        return int(_df.iloc[0]["Time"]) if len(_df) else -1

    while True:
        RUNNING.wait()
        try:
            if session is None:
                session = Session(IOTDB_HOST, IOTDB_PORT, IOTDB_USER, IOTDB_PASS)
                session.open()
                print("采集线程：已连接 IoTDB")
                baseline_ts = probe_latest_ts(session)   # 按下开始后的最新 ts
                last_emitted_ts = None
                next_candidate_ts = None
                gap_since_ms = None
                gap_expected_ts = None
                last_real_feats = None
                staging.clear()
                print(f"采集线程：baseline_ts = {baseline_ts}")

            # 拉 baseline 之后的新数据（升序）
            sql = (
                f"SELECT {sel} FROM {IOTDB_DB} "
                f"WHERE Time > {baseline_ts} "
                f"ORDER BY Time ASC LIMIT {QUERY_LIMIT}"
            )
            dataset = session.execute_query_statement(sql)
            df = dataset.todf()

            if len(df) == 0:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # 合并入 staging
            batch_max_ts = -1
            for _, row in df.iterrows():
                ts = int(row["Time"])
                b = staging.get(ts) or {}

                v_water   = getv(row, COL_WATER)
                v_inair_t = getv(row, COL_INAIR_T)
                v_exh_t   = getv(row, COL_EXH_T)
                v_inair_p = getv(row, COL_INAIR_P)
                v_o2      = getv(row, COL_O2)

                if v_water   is not None and not pd.isna(v_water):   b[COL_WATER]   = float(v_water)
                if v_inair_t is not None and not pd.isna(v_inair_t): b[COL_INAIR_T] = float(v_inair_t)
                if v_exh_t   is not None and not pd.isna(v_exh_t):   b[COL_EXH_T]   = float(v_exh_t)
                if v_inair_p is not None and not pd.isna(v_inair_p): b[COL_INAIR_P] = float(v_inair_p)
                if v_o2      is not None and not pd.isna(v_o2):      b[COL_O2]      = float(v_o2)

                staging[ts] = b
                if ts > batch_max_ts:
                    batch_max_ts = ts

            # 忽略最新5行 → 计算水印
            watermark_ts = batch_max_ts - TAIL_IGNORE_ROWS * EXPECT_DT_MS if batch_max_ts >= 0 else -1

            # === DEBUG(3)：水印/候选概览 ===
            if watermark_ts >= 0:
                candidates_cnt = sum(1 for t in staging.keys() if t <= watermark_ts)
                print(f"[DEBUG][Watermark] watermark={watermark_ts} | candidates<=wm={candidates_cnt} | staging={len(staging)}")

            # 初始化候选 ts
            if next_candidate_ts is None:
                candidates = [t for t in staging.keys() if t <= watermark_ts]
                next_candidate_ts = min(candidates) if candidates else None

            emitted = 0
            # 发射：必须 行齐、<=watermark、严格10ms连续（若超时则补帧）
            while next_candidate_ts is not None and next_candidate_ts <= watermark_ts:
                b = staging.get(next_candidate_ts)
                if b is None:
                    candidates = [t for t in staging.keys() if t > next_candidate_ts and t <= watermark_ts]
                    next_candidate_ts = min(candidates) if candidates else None
                    continue

                # === DEBUG(1)：行不完整 → 打印并等待 ===
                if not all(c in b for c in need_cols):
                    print(f"[DEBUG][IncompleteRow] ts={next_candidate_ts} cols={list(b.keys())} 等待补齐...")
                    break

                # 连续性判断（支持缺帧前向补齐）
                if last_emitted_ts is not None:
                    expected = last_emitted_ts + EXPECT_DT_MS
                    if next_candidate_ts != expected:
                        now_ms = int(time.time()*1000)
                        # 初次或目标变化 → 开始/重置计时
                        if gap_expected_ts != expected:
                            gap_expected_ts = expected
                            gap_since_ms = now_ms

                        waited = now_ms - (gap_since_ms or now_ms)
                        print(f"[DEBUG][GapWait] 期待 ts={expected}, 实际 ts={next_candidate_ts}，已等待 {waited}ms...")

                        # 等待没到阈值 → 继续等
                        if waited < GAP_MAX_WAIT_MS:
                            break

                        # 超时：前向补帧（用最近一次真实帧 last_real_feats）
                        fill_cnt = 0
                        if last_real_feats is not None:
                            fill_ts = expected
                            while fill_ts < next_candidate_ts and fill_cnt < GAP_FILL_MAX_STEPS:
                                feats_fill = last_real_feats.copy()
                                try:
                                    cache_queue.put((fill_ts, feats_fill), timeout=0.2)
                                    with pc_lock:
                                        global put_count
                                        put_count += 1
                                except queue.Full:
                                    print(f"[DEBUG][QueueFull] cache_queue 已满({cache_queue.qsize()}条)，填充帧等待消费者...")
                                    time.sleep(0.05)
                                    continue

                                last_emitted_ts = fill_ts
                                baseline_ts = max(baseline_ts, last_emitted_ts)
                                emitted += 1
                                fill_ts += EXPECT_DT_MS
                                fill_cnt += 1

                            if fill_cnt > 0:
                                print(f"[FILL][GapFF] 用上一帧前向填充 {fill_cnt} 帧，范围 [{expected} ~ {fill_ts-EXPECT_DT_MS}]")
                                # 清空gap等待，继续处理（这时 expected 会追上来）
                                gap_since_ms = None
                                gap_expected_ts = None
                                # 继续 while，不 break，让循环继续检查后续候选
                                continue
                            else:
                                # 没能填（队列满等情况），下轮再试
                                break
                        else:
                            # 还没发过真实帧，无基准可填，只能等待
                            break

                # 走到这里：要么连续、要么补帧后已连续 → 正常发射真实帧
                oil = o2_to_oil_flow(b[COL_O2])
                feats = np.array([float(oil), b[COL_WATER], b[COL_EXH_T], b[COL_INAIR_P], b[COL_INAIR_T]],
                                 dtype=np.float32)

                try:
                    cache_queue.put((next_candidate_ts, feats), timeout=0.2)
                    with pc_lock:
                        put_count += 1
                except queue.Full:
                    print(f"[DEBUG][QueueFull] cache_queue 已满({cache_queue.qsize()}条)，等待消费者...")
                    time.sleep(0.05)
                    continue

                # 记住最近一帧真实特征，供后续补帧使用
                last_real_feats = feats.copy()

                last_emitted_ts = next_candidate_ts
                baseline_ts = max(baseline_ts, last_emitted_ts)
                emitted += 1

                # 清理推进
                staging.pop(next_candidate_ts, None)
                candidates = [t for t in staging.keys() if t > last_emitted_ts and t <= watermark_ts]
                next_candidate_ts = min(candidates) if candidates else None

            if emitted:
                print(f"[Fetch] emitted={emitted}, cache_q={cache_queue.qsize()}, "
                      f"baseline_ts={baseline_ts}, watermark_ts={watermark_ts}")

            # 降速空转
            if len(df) < QUERY_LIMIT and emitted == 0:
                time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            print(f"[采集线程] 异常：{e}，{RETRY_WAIT_SEC}s后重试连接...")
            try:
                if session: session.close()
            except Exception:
                pass
            session = None
            time.sleep(RETRY_WAIT_SEC)

# ================= 预测线程（非重叠6帧窗口） =================
def predict_thread():
    """
    每次从 cache_queue 取6条（严格10ms连续/或补帧后连续），组成 (1,6,5) 推理；用完即丢，不复用
    """
    print("推理线程启动（非重叠6帧窗口）...")

    model = SimpleCNN_LSTM(input_size=5, hidden_size=32, dropout=0.3).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        RUNNING.wait()

        # 取满6条
        ts_list = []
        feats_list = []
        while len(ts_list) < WINDOW_LEN:
            RUNNING.wait()
            ts, feats = cache_queue.get()
            with gc_lock:
                global get_count
                get_count += 1
            ts_list.append(ts)
            feats_list.append(feats.astype(np.float32))

        # 安全校验连续性（包含补帧后）
        ok = True
        for i in range(1, WINDOW_LEN):
            if ts_list[i] - ts_list[i-1] != EXPECT_DT_MS:
                ok = False; break
        if not ok:
            print(f"[Predict][WARN] 窗口时间不连续，丢弃本窗口: {ts_list}")
            ts_list.clear(); feats_list.clear()
            continue

        # (1,6,5)
        arr  = np.stack(feats_list, axis=0)
        arrn = np.stack([normalize_feat(x) for x in arr], axis=0)
        x    = torch.tensor(arrn, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            y_norm = model(x).view(-1).detach().cpu().numpy()[0]
        y_real = float(denormalize_eff(np.array([y_norm]))[0])

        # 输出（取窗口最后一帧的原始值）
        oil, water, exh, inair_p, inair_t = feats_list[-1].tolist()
        power_w = calc_power_w(oil_flow_gps=oil, efficiency_percent=y_real)

        pred_ts = ts_list[-1] + EXPECT_DT_MS  # 前6→预第7拍
        pred_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pred_ts / 1000.0))

        item = {
            "Time": pred_time_str,
            "Prediction": round(y_real, 2),
            "Power": round(power_w, 2),
            "water_temp": round(water, 2),
            "in_air_temp": round(inair_t, 2),
            "exh_temp": round(exh, 2),
            "in_air_pressure": round(inair_p, 2),
            "oil_flow": round(oil, 2),
        }
        ui_queue.put(item)  # 若担心 UI 堵塞，可换 safe_put 方案
        _write_single_csv(item)

        _batch_cache.append([item["Time"], f"{item['Prediction']:.2f}"])
        if len(_batch_cache) >= BATCH_FLUSH_N:
            _write_batch_csv_flush()

        span = ts_list[-1] - ts_list[0]
        print(f"[Predict] window {ts_list[0]}~{ts_list[-1]} (span={span}ms) | cache_q={cache_queue.qsize()}")

        ts_list.clear()
        feats_list.clear()

# ================= 监控线程（真实吞吐） =================
def monitor_thread():
    last_put = 0
    last_get = 0
    tick = 0
    while True:
        time.sleep(1.0)
        with pc_lock: cur_put = put_count
        with gc_lock: cur_get = get_count
        in_rate  = cur_put - last_put   # 条/秒
        out_rate = cur_get - last_get   # 条/秒
        print(f"[Monitor {tick:04d}] cache_q={cache_queue.qsize()} | ui_q={ui_queue.qsize()} "
              f"| in={in_rate}/s | out={out_rate}/s | running={RUNNING.is_set()}")
        last_put, last_get = cur_put, cur_get
        tick += 1

# ================= UI 控制 =================
def build_controls(root: tk.Tk):
    top = tk.Frame(root); top.pack(side="top", fill="x", padx=10, pady=6)

    def set_running():
        RUNNING.set(); print("[Control] 已开始")

    def set_paused():
        RUNNING.clear(); print("[Control] 已暂停")

    btn_start = tk.Button(top, text="开始 (r)", command=set_running)
    btn_pause = tk.Button(top, text="暂停 (p)", command=set_paused)
    btn_start.pack(side="left"); btn_pause.pack(side="left", padx=8)

    root.bind("r", lambda e: set_running())
    root.bind("p", lambda e: set_paused())

# ================= 主入口 =================
if __name__ == "__main__":
    t_fetch = threading.Thread(target=iotdb_fetch_thread, daemon=True)
    t_pred  = threading.Thread(target=predict_thread,    daemon=True)
    t_mon   = threading.Thread(target=monitor_thread,    daemon=True)
    t_fetch.start(); t_pred.start(); t_mon.start()

    print("主程序运行中（默认暂停，按“开始”或按键 r 开始）...")
    app = RealtimeUI(ui_queue, init_interval_ms=400, max_rows=200)
    build_controls(app.root)
    app.run()
