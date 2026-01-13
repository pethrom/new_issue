# -*- coding: utf-8 -*-
"""
IoTDB 增量拉取 + 忽略最新5行 + 行完整 + 严格10ms + 单帧BP前向(效率&功率) + Tk UI
调试输出：
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
import tkinter as tk
from iotdb.Session import Session
from scipy.io import loadmat

# === 项目内模块（保留） ===
from ui_realtime import RealtimeUI

# ================= 基本配置 =================
EXPECT_DT_MS = 10             # 期望时间步长（严格10ms）
TAIL_IGNORE_ROWS = 5          # 忽略最新5行（≈50ms）
QUERY_LIMIT = 1024
POLL_INTERVAL_SEC = 0.3
RETRY_WAIT_SEC = 2.0

# --- 缺帧处理（前向补帧） ---
GAP_MAX_WAIT_MS = 500         # 等待缺帧的最大时长；超过则开始补
GAP_FILL_MAX_STEPS = 24       # 一次最多补多少帧（240ms）

# IoTDB
IOTDB_HOST = "127.0.0.1"
IOTDB_PORT = "6667"
IOTDB_USER = "root"
IOTDB_PASS = "root"
IOTDB_DB   = "root.engine1"

# 列映射（与你原代码一致）
COL_WATER   = "EngT1_degC"   # 水温
COL_INAIR_T = "InAirT_degC"  # 进气温度
COL_EXH_T   = "ExhOutT_degC" # 排气温度
COL_INAIR_P = "InAirP_kPa"   # 进气压力
COL_O2      = "O2_percent"   # O2%
IOTDB_SELECT_COLUMNS = [COL_WATER, COL_INAIR_T, COL_EXH_T, COL_INAIR_P, COL_O2]

# === 两份 BP 权重（效率 / 功率） ===
BP_MAT_PATH_EFF = "bp_xiaolv_weights_V4_0.mat"      # 效率模型（输出单位：%）
BP_MAT_PATH_PWR = "bp_gonglv_weights_V4_0.mat"    # 功率模型（输出单位：W）

# ================= MATLAB BP 前向（tansig→purelin + mapminmax） =================
def tansig(x: np.ndarray) -> np.ndarray:
    return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

def _struct_to_ps(st) -> dict:
    """将 MATLAB struct/dtype 转为 {'xoffset': (F,), 'gain': (F,), 'ymin': float}（兼容多种形态）"""
    if isinstance(st, dict):
        xoffset = np.array(st['xoffset']).reshape(-1)
        gain    = np.array(st['gain']).reshape(-1)
        ymin    = float(np.array(st['ymin']).reshape(-1)[0])
        return {'xoffset': xoffset, 'gain': gain, 'ymin': ymin}
    if isinstance(st, np.ndarray) and st.dtype == np.object_:
        st = st.item()
    try:
        xoffset = np.array(st['xoffset']).reshape(-1)
        gain    = np.array(st['gain']).reshape(-1)
        ymin    = float(np.array(st['ymin']).reshape(-1)[0])
        return {'xoffset': xoffset, 'gain': gain, 'ymin': ymin}
    except Exception:
        xoffset = np.array(getattr(st, 'xoffset')).reshape(-1)
        gain    = np.array(getattr(st, 'gain')).reshape(-1)
        ymin    = float(np.array(getattr(st, 'ymin')).reshape(-1)[0])
        return {'xoffset': xoffset, 'gain': gain, 'ymin': ymin}

def mapminmax_apply(X: np.ndarray, ps: dict) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    xoffset = ps['xoffset'].reshape(1, -1)
    gain    = ps['gain'].reshape(1, -1)
    ymin    = float(ps['ymin'])
    return (X - xoffset) * gain + ymin

def mapminmax_reverse(Y: np.ndarray, ps: dict) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64)
    xoffset = ps['xoffset'].reshape(1, -1)
    gain    = ps['gain'].reshape(1, -1)
    ymin    = float(ps['ymin'])
    return (Y - ymin) / gain + xoffset

def _as_int(x) -> int:
    """稳健转 int：兼容 python int / numpy 标量 / 小数组"""
    import numpy as _np
    if isinstance(x, int):
        return x
    try:
        return int(_np.array(x).reshape(()).item())
    except Exception:
        return int(x)

class MatlabBP:
    """单隐层 BP：输入F→隐层H(tansig)→输出O(purelin)，数值域由 MATLAB 的 mapminmax 决定。"""
    def __init__(self, mat_path: str, strict_shape_check: bool = True):
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"未找到权重文件：{mat_path}")
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        # 原始读取
        self.W1 = np.array(m['W1'], dtype=np.float64)                 # (H, F) 期望
        self.B1 = np.array(m['B1'], dtype=np.float64)                 # (H,) / (1,H) / (H,1)
        self.W2 = np.array(m['W2'], dtype=np.float64)                 # (O, H) / (H, O) / 1D
        self.B2 = np.array(m['B2'], dtype=np.float64)                 # (O,) / (1,O) / (O,1)

        self.in_ps  = _struct_to_ps(m['in_ps'])
        self.out_ps = _struct_to_ps(m['out_ps'])

        # —— 以 W1/B2 权威确定尺寸，并自适应修正 W2/B1/B2 形状 ——
        if self.W1.ndim != 2:
            raise ValueError(f"W1 维度异常：{self.W1.shape}")
        H, F = self.W1.shape

        self.B1 = self.B1.reshape(1, -1)
        self.B2 = self.B2.reshape(1, -1)
        O = self.B2.shape[1]  # 输出维度以 B2 权威确定

        W2 = self.W2
        if W2.ndim == 1:
            if W2.size == O * H:
                W2 = W2.reshape(O, H)
            else:
                raise ValueError(f"W2 维度异常：flat size={W2.size}, 但 H={H}, O={O}")
        elif W2.shape == (O, H):
            pass
        elif W2.shape == (H, O):
            W2 = W2.T
        else:
            if (H in W2.shape) and (O in W2.shape) and W2.ndim == 2:
                axis_h = int(W2.shape.index(H))
                axis_o = 1 - axis_h
                W2 = np.transpose(W2, (axis_o, axis_h))
            else:
                raise ValueError(f"W2 形状不匹配：{W2.shape}，期望 (O,H)=({O},{H}) 或 (H,O)=({H},{O})")
        self.W2 = W2

        # 尺寸
        self.input_size  = F
        self.hidden_size = H
        self.output_size = O

        if strict_shape_check:
            assert self.W1.shape == (self.hidden_size, self.input_size),  "W1 形状与尺寸不符"
            assert self.W2.shape == (self.output_size, self.hidden_size), "W2 形状与尺寸不符"
            assert self.B1.shape == (1, self.hidden_size),                "B1 形状错误"
            assert self.B2.shape == (1, self.output_size),                "B2 形状错误"

        # 可选：保存的列索引（1-based），仅用于日志或自检
        self.input_cols_1based  = m.get('input_cols_1based', None)
        self.output_cols_1based = m.get('output_cols_1based', None)

        print(f"[BP:{os.path.basename(mat_path)}] Shapes: W1={self.W1.shape}, B1={self.B1.shape}, "
              f"W2={self.W2.shape}, B2={self.B2.shape}  => F={F}, H={H}, O={O}")

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == self.input_size, \
            f"输入维度为 {X.shape}, 模型期望特征数 {self.input_size}"
        Xn = mapminmax_apply(X, self.in_ps)
        A1 = tansig(Xn @ self.W1.T + self.B1)   # (N,H)
        A2 = A1 @ self.W2.T + self.B2           # (N,O)
        Y  = mapminmax_reverse(A2, self.out_ps) # (N,O)
        return Y

    def predict_one(self, feats) -> float:
        X = np.array(feats, dtype=np.float64).reshape(1, -1)
        y = self.predict_batch(X)
        return float(y.reshape(-1)[0])

# ================= 物理量计算 =================
def o2_to_oil_flow(o2_percent: float) -> float:
    """O2% → 油量(g/s)（与你原代码一致）"""
    Q2 = float(o2_percent)
    denom = 0.2095 - Q2 * 0.01
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0 else -1e-6
    U2 = (0.2098 + 0.2737 * Q2 * 0.01) / denom
    if U2 <= 1e-6:
        U2 = 1e-6
    oil = 17.0 / (U2 * 14.8)
    return max(float(oil), 0.0)

# ================= 队列 / 控制 / CSV =================
cache_queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(maxsize=50000)
ui_queue:    "queue.Queue[dict]" = queue.Queue(maxsize=5000)

RUNNING = threading.Event()
RUNNING.clear()

put_count = 0
get_count = 0
pc_lock = threading.Lock()
gc_lock = threading.Lock()

BATCH_CSV_PATH = "online_predict_results.csv"
BATCH_FLUSH_N = 200
_batch_cache = []

SINGLE_CSV_PATH = "BPpredict_log.csv"
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
                "Time",
                "Prediction(%)",   # 效率
                "Power(kW)",       # ✅ 改为 kW，保持与你的 UI 展示一致
                "water_temp", "in_air_temp", "exh_temp",
                "in_air_pressure", "oil_flow(g/s)"
            ])

def _write_single_csv(item: dict):
    _ensure_single_csv()
    _single_csv_writer.writerow([
        item["Time"],
        f"{item['Prediction']:.2f}",
        f"{item['Power']:.3f}",   # kW 显示到 3 位小数更直观
        f"{item['water_temp']:.2f}",
        f"{item['in_air_temp']:.2f}",
        f"{item['exh_temp']:.2f}",
        f"{item['in_air_pressure']:.2f}",
        f"{item['oil_flow']:.3f}",
    ])
    _single_csv_file.flush()

def _write_batch_csv_flush():
    if not _batch_cache:
        return
    df = pd.DataFrame(_batch_cache, columns=["Time", "Prediction"])
    header = not os.path.exists(BATCH_CSV_PATH)
    df.to_csv(BATCH_CSV_PATH, mode="a", header=header, index=False, encoding="utf-8-sig")
    _batch_cache.clear()

# ================= 采集线程（含水印/补帧/调试） =================
def iotdb_fetch_thread():
    print("采集线程启动：baseline对齐 + 忽略最新5行 + 行完整 + 严格10ms + 缺帧前向补帧 + 大缓存队列")

    sel = ", ".join(IOTDB_SELECT_COLUMNS)
    session = None
    staging: Dict[int, Dict[str, float]] = {}

    baseline_ts = -1
    last_emitted_ts = None
    next_candidate_ts = None

    gap_since_ms = None
    gap_expected_ts = None
    last_real_feats = None

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
                baseline_ts = probe_latest_ts(session)
                last_emitted_ts = None
                next_candidate_ts = None
                gap_since_ms = None
                gap_expected_ts = None
                last_real_feats = None
                staging.clear()
                print(f"采集线程：baseline_ts = {baseline_ts}")

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

            watermark_ts = batch_max_ts - TAIL_IGNORE_ROWS * EXPECT_DT_MS if batch_max_ts >= 0 else -1

            if watermark_ts >= 0:
                candidates_cnt = sum(1 for t in staging.keys() if t <= watermark_ts)
                print(f"[DEBUG][Watermark] watermark={watermark_ts} | candidates<=wm={candidates_cnt} | staging={len(staging)}")

            if next_candidate_ts is None:
                candidates = [t for t in staging.keys() if t <= watermark_ts]
                next_candidate_ts = min(candidates) if candidates else None

            emitted = 0
            while next_candidate_ts is not None and next_candidate_ts <= watermark_ts:
                b = staging.get(next_candidate_ts)
                if b is None:
                    candidates = [t for t in staging.keys() if t > next_candidate_ts and t <= watermark_ts]
                    next_candidate_ts = min(candidates) if candidates else None
                    continue

                if not all(c in b for c in need_cols):
                    print(f"[DEBUG][IncompleteRow] ts={next_candidate_ts} cols={list(b.keys())} 等待补齐...")
                    break

                if last_emitted_ts is not None:
                    expected = last_emitted_ts + EXPECT_DT_MS
                    if next_candidate_ts != expected:
                        now_ms = int(time.time()*1000)
                        if gap_expected_ts != expected:
                            gap_expected_ts = expected
                            gap_since_ms = now_ms

                        waited = now_ms - (gap_since_ms or now_ms)
                        print(f"[DEBUG][GapWait] 期待 ts={expected}, 实际 ts={next_candidate_ts}，已等待 {waited}ms...")

                        if waited < GAP_MAX_WAIT_MS:
                            break

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
                                gap_since_ms = None
                                gap_expected_ts = None
                                continue
                            else:
                                break
                        else:
                            break

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

                last_real_feats = feats.copy()

                last_emitted_ts = next_candidate_ts
                baseline_ts = max(baseline_ts, last_emitted_ts)
                emitted += 1

                staging.pop(next_candidate_ts, None)
                candidates = [t for t in staging.keys() if t > last_emitted_ts and t <= watermark_ts]
                next_candidate_ts = min(candidates) if candidates else None

            if emitted:
                print(f"[Fetch] emitted={emitted}, cache_q={cache_queue.qsize()}, "
                      f"baseline_ts={baseline_ts}, watermark_ts={watermark_ts}")

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

# ================= 预测线程（双 BP：效率 + 功率） =================
def predict_thread():
    print("推理线程启动（双 BP：效率 + 功率）...")
    # 明确：eff = 效率模型（%），pwr = 功率模型（W）
    bp_eff = MatlabBP(BP_MAT_PATH_EFF)   # 输出单位：%
    bp_pwr = MatlabBP(BP_MAT_PATH_PWR)   # 输出单位：W

    # 运行时采集的特征原始顺序（feats = [油量, 水温, 排气温度, 进气压力, 进气温度]）
    # 你要求送入模型的新顺序：
    #   [进气压力, 排气温度, 进气温度, 水温, 油量]
    RUNTIME_TO_MODEL = lambda oil, water, exh, inair_p, inair_t: [inair_p, exh, inair_t, water, oil]

    while True:
        RUNNING.wait()

        ts, feats = cache_queue.get()
        with gc_lock:
            global get_count
            get_count += 1

        # 原始 feats 顺序: [油量, 水温, 排气温度, 进气压力, 进气温度]
        oil, water, exh, inair_p, inair_t = feats.tolist()

        # 组装给模型的输入向量（按你最新要求的顺序）
        x = RUNTIME_TO_MODEL(oil, water, exh, inair_p, inair_t)

        # 各自模型前向（内部含归一化/反归一化）
        y_eff_percent = float(bp_eff.predict_one(x))   # %
        y_pwr_watt    = float(bp_pwr.predict_one(x))   # W

        # 传给 UI 的 Power 必须是 W，UI 自己 /1000 显示 kW
        y_eff_percent = max(y_eff_percent, 0.0)  # 兜底不为负
        y_pwr_watt = max(y_pwr_watt, 0.0)  # 兜底不为负

        pred_ts = ts + EXPECT_DT_MS
        pred_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pred_ts / 1000.0))

        item = {
            "Time": pred_time_str,
            "Prediction": round(y_eff_percent, 2),  # %
            "Power": round(y_pwr_watt, 2),          # ✅ W（UI 会 /1000 显示 kW）
            "water_temp": round(water, 2),
            "in_air_temp": round(inair_t, 2),
            "exh_temp": round(exh, 2),
            "in_air_pressure": round(inair_p, 2),
            "oil_flow": round(oil, 3),
        }
        ui_queue.put(item)
        _write_single_csv(item)

        _batch_cache.append([item["Time"], f"{item['Prediction']:.2f}"])
        if len(_batch_cache) >= BATCH_FLUSH_N:
            _write_batch_csv_flush()

        print(f"[Predict] ts={ts} → eff={item['Prediction']}% | pwr={item['Power']} W | cache_q={cache_queue.qsize()}")

# ================= 监控线程 =================
def monitor_thread():
    last_put = 0
    last_get = 0
    tick = 0
    while True:
        time.sleep(1.0)
        with pc_lock: cur_put = put_count
        with gc_lock: cur_get = get_count
        in_rate  = cur_put - last_put
        out_rate = cur_get - last_get
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
