# ui_realtime.py
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import time
from collections import deque
import os
import csv

# 表格显示列（保持你之前的顺序与命名）
COLS = [
    "Time", "Prediction",          # 时间戳、热效率（数值；表格不加 %）
    "oil_flow",                    # 油量
    "water_temp",                  # 水温
    "exh_temp",                    # 排气温度
    "in_air_pressure",             # 进气压力
    "in_air_temp"                  # 进气温度
]

class RealtimeUI:
    """
    轻量级 Tk 实时界面：
    - 顶部：显示“当前预测热效率(%)”与“当前预测功率(kW)”
      * 显示为“过去 window_ms 内的平均值”（默认 1s）
    - 中部：表格展示最近的多行输入/输出（插入最新一条）
    - 额外：每 1s 将窗口均值追加写入 CSV (BPpredictP1s.csv)
    """
    def __init__(self, ui_queue, init_interval_ms=1000, max_rows=200,
                 title="CNN-LSTM 在线预测", window_ms=1000,
                 write_avg_csv=True, avg_csv_path="BPpredictP1s.csv"):
        self.ui_queue = ui_queue
        self.update_interval_ms = int(init_interval_ms)
        self.max_rows = int(max_rows)

        # 最新单条，供表格插入
        self.buffer_latest = None
        self.last_shown_ts = None  # 表格插入去重（基于 Time）

        # === 1 秒滑动窗口缓存：[(ts_ms, rec), ...] ===
        self.window_ms = int(window_ms)
        self.window = deque()

        # === 聚合结果 CSV ===
        self.write_avg_csv = bool(write_avg_csv)
        self.avg_csv_path = avg_csv_path
        self._last_avg_written_wallclock_sec = None  # 控制每秒写一次
        if self.write_avg_csv:
            self._ensure_avg_csv()

        # ==== Tk 基本窗口 ====
        self.root = tk.Tk()
        self.root.title(title)

        # ---- 顶部显示区：热效率 + 功率 ----
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="当前预测热效率(1s均值)：", font=("Segoe UI", 12)).pack(side="left")
        self.pred_var = tk.StringVar(value="--")
        ttk.Label(top, textvariable=self.pred_var, font=("Segoe UI", 18, "bold")).pack(side="left", padx=8)

        ttk.Label(top, text="当前预测功率(1s均值)：", font=("Segoe UI", 12)).pack(side="left", padx=(24, 4))
        self.power_var = tk.StringVar(value="--")
        ttk.Label(top, textvariable=self.power_var, font=("Segoe UI", 18, "bold")).pack(side="left", padx=8)

        # ---- 表格区域 ----
        mid = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(mid, columns=COLS, show="headings", height=12)
        for col in COLS:
            self.tree.heading(col, text=col)
        col_widths = {
            "Time": 160, "Prediction": 100,
            "oil_flow": 100, "water_temp": 100, "exh_temp": 110,
            "in_air_pressure": 110, "in_air_temp": 110
        }
        for col in COLS:
            self.tree.column(col, width=col_widths.get(col, 100), anchor="center", stretch=True)

        yscroll = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=yscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")

        # ---- 状态栏 ----
        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        self.status_var = tk.StringVar(value=f"刷新间隔：{self.update_interval_ms} ms | 聚合窗口：{self.window_ms} ms")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left")

        # 定时任务：轮询队列 + 渲染
        self.root.after(self.update_interval_ms, self._poll_queue)
        self.root.after(self.update_interval_ms, self._render_tick)

    # ====== 公开方法 ======
    def run(self):
        self.root.mainloop()

    def set_update_interval(self, interval_ms: int):
        self.update_interval_ms = max(100, int(interval_ms))
        self.status_var.set(f"刷新间隔：{self.update_interval_ms} ms | 聚合窗口：{self.window_ms} ms")

    # ====== 内部方法 ======
    def _ensure_avg_csv(self):
        need_header = not os.path.exists(self.avg_csv_path)
        # 直接用追加模式，UTF-8-SIG 便于 Excel
        with open(self.avg_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow(["Time", "AvgPrediction(%)", "AvgPower(kW)", "Count"])

    def _append_avg_csv(self, when_str: str, avg_pred: float, avg_kw: float, count: int):
        if not self.write_avg_csv:
            return
        with open(self.avg_csv_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([when_str, f"{avg_pred:.4f}", f"{avg_kw:.4f}", count])

    def _poll_queue(self):
        """
        轮询 ui_queue：收集本周期内所有预测记录，入 1s 滑动窗口（不丢样）。
        同时保留最后一条用于表格插入。
        """
        now_ms = lambda: int(time.time() * 1000)
        got_any = False

        try:
            while True:
                rec = self.ui_queue.get_nowait()
                if isinstance(rec, dict) and ("Prediction" in rec or rec.get("__type__") == "pred"):
                    t = now_ms()  # 以到达 UI 的时刻为准做 1s 窗口
                    self.window.append((t, rec))
                    self.buffer_latest = rec
                    got_any = True
        except Exception:
            pass

        # 清理超过窗口期的数据
        if self.window:
            t_now = now_ms()
            while self.window and (t_now - self.window[0][0] > self.window_ms):
                self.window.popleft()

        # 状态提示
        if got_any and self.buffer_latest is not None:
            ts = self.buffer_latest.get("Time", "--")
            self.status_var.set(
                f"刷新间隔：{self.update_interval_ms} ms | 聚合窗口：{self.window_ms} ms | "
                f"窗口内样本数：{len(self.window)} | 最近预测时间：{ts}"
            )

        self.root.after(self.update_interval_ms, self._poll_queue)

    def _render_tick(self):
        """
        顶部：显示 1s 窗口内的均值（效率%、功率kW）
        表格：遇到新时间戳插入一行“最新样本”（不做均值，以便看原始细节）
        同时：每秒将均值追加写入 CSV（避免重复写）
        """
        # ---- 顶部 1s 均值 ----
        avg_pred, avg_kw, count = None, None, 0
        if self.window:
            preds = []
            powers_w = []
            for _, rec in list(self.window):
                try:
                    preds.append(float(rec.get("Prediction", None)))
                except Exception:
                    pass
                try:
                    p_w = rec.get("Power", None)  # 底层 Power 单位是 W
                    if p_w is not None:
                        powers_w.append(float(p_w))
                except Exception:
                    pass

            count = max(len(preds), len(powers_w))
            if preds:
                avg_pred = sum(preds) / len(preds)
                self.pred_var.set(f"{avg_pred:.4f}%")
            else:
                self.pred_var.set("--")

            if powers_w:
                avg_kw = (sum(powers_w) / len(powers_w)) / 1000.0
                self.power_var.set(f"{avg_kw:.2f} kW")
            else:
                self.power_var.set("--")
        else:
            self.pred_var.set("--")
            self.power_var.set("--")

        # ---- 每秒写一次 CSV（有数据才写）----
        if avg_pred is not None or avg_kw is not None:
            wallclock_sec = int(time.time())  # 秒级时间戳
            if wallclock_sec != self._last_avg_written_wallclock_sec:
                self._last_avg_written_wallclock_sec = wallclock_sec
                when_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wallclock_sec))
                self._append_avg_csv(when_str, avg_pred if avg_pred is not None else float("nan"),
                                     avg_kw if avg_kw is not None else float("nan"),
                                     count)

        # ---- 表格：只插最新一条 ----
        if self.buffer_latest is not None:
            rec = self.buffer_latest
            ts = rec.get("Time", None)
            if ts is not None and ts != self.last_shown_ts:
                oil, water, exh, inair_p, inair_t = self._extract_ordered_values(rec)
                try:
                    self.tree.insert(
                        "",
                        0,
                        values=(ts, rec.get("Prediction", "--"),
                                oil, water, exh, inair_p, inair_t),
                    )
                    self.last_shown_ts = ts
                    if len(self.tree.get_children()) > self.max_rows:
                        self.tree.delete(self.tree.get_children()[-1])
                except Exception:
                    pass

        self.root.after(self.update_interval_ms, self._render_tick)

    def _extract_ordered_values(self, rec: dict):
        """返回顺序：oil_flow, water_temp, exh_temp, in_air_pressure, in_air_temp"""
        def _get(k):
            v = rec.get(k, None)
            if v is None:
                return "--"
            try:
                return float(v)
            except Exception:
                return v
        oil = _get("oil_flow")
        water = _get("water_temp")
        exh = _get("exh_temp")
        inair_p = _get("in_air_pressure")
        inair_t = _get("in_air_temp")
        return oil, water, exh, inair_p, inair_t


# 独立测试（可选）
if __name__ == "__main__":
    import queue, threading, random
    q = queue.Queue()
    ui = RealtimeUI(q, init_interval_ms=1000, max_rows=200, window_ms=1000,
                    write_avg_csv=True, avg_csv_path="BPpredictP1s.csv")

    def feeder():
        while True:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            eff = 20 + 5 * random.random()
            power_w = 5000 + 2000 * random.random()  # W
            rec = {
                "__type__": "pred",
                "Time": now,
                "Prediction": eff,
                "Power": power_w,  # 传 W；UI/CSV 都写 kW
                "oil_flow": 5 + 2 * random.random(),
                "water_temp": 80 + 10 * random.random(),
                "exh_temp": 300 + 50 * random.random(),
                "in_air_pressure": 120 + 10 * random.random(),
                "in_air_temp": 40 + 5 * random.random(),
            }
            q.put(rec)
            time.sleep(0.02)  # 约 50Hz
    threading.Thread(target=feeder, daemon=True).start()
    ui.run()
