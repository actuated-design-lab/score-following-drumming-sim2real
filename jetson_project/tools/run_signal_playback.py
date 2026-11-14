"""
Porcaro Robot: 指令信号CSVの再生（ポリシー無し）(Async: Recv 200Hz / Send 50Hz)
Target: Jetson Orin Nano + MicroLabBox

入力は signals/<名前>.csv。signals/<名前>_guide.csv（列 time, message）があれば、その時刻に操作の指示を表示する。実測（200Hz）は data/<user>/<venue>/playback_<日付>/data_<名前>_<unixtime>.csv に保存する。
user / venue は --user / --venue か環境変数 PORCARO_USER / PORCARO_VENUE（src/repo_paths.py）。

python3 tools/run_signal_playback.py exp2_step_response --user user0 --venue jfps2026
python3 tools/run_signal_playback.py pm_ramp            # PORCARO_USER / PORCARO_VENUE を export 済みなら
"""
import serial
import struct
import json
import time
import numpy as np
import pandas as pd
import threading
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from repo_paths import (add_user_venue_args, require_user_venue,  # noqa: E402
                        run_dir, signal_path)
from pressure_slots import SLOTS, named  # noqa: E402

# System Configuration
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 230400
CONTROL_DT = 0.02        # 制御・送信は50Hz

SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [Meas_DF, Meas_F, Meas_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7
SYNC_SKIP_PACKETS = 5    # 同期の直後のこのパケット数（25 ms）までの読み飛ばしは「同期のくせ」として別に数える

class SensorReceiver(threading.Thread):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.sensor_logs = []
        self.clear_flag = False
        self.lock = threading.Lock()
        self.daemon = True
        # 受信の統計（deploy_policy の json と同じ項目。同期（GO）のあとから数える。2026-11-03、QC を RL と FF でそろえるため）
        self.n_dropped = 0
        # 同期（GO）の瞬間に受信のバッファを途中で捨てるので、その直後に半端なパケットを読み飛ばす 1 回（0〜57 バイト）が必ず起きる。
        # これは通信の乱れではないので、同期のあと最初の SYNC_SKIP_PACKETS パケットまでの読み飛ばしは別に数える（2026-11-04）
        self.n_dropped_sync = 0
        self.n_after_clear = 0

    def run(self):
        self.ser.reset_input_buffer()
        buffer = b''

        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)

                    while len(buffer) >= RECV_PACKET_LEN:
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            self.n_dropped += max(0, len(buffer) - 1)
                            buffer = buffer[-1:]
                            break
                        if idx > 0:
                            self.n_dropped += idx
                            buffer = buffer[idx:]
                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        packet = buffer[:RECV_PACKET_LEN]
                        buffer = buffer[RECV_PACKET_LEN:]
                        self._process_packet(packet[2:])
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[Receiver Error] {e}")
                self.running = False

    def _process_packet(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            sample = {
                'timestamp_pc': time.perf_counter(), # 参考（USBジッタ確認用）
                **named(data[0], data[1], data[2]),  # 枠 → 筋の名前（config/pressure_slots.yaml）
                'pres_mapped': 1,
                'wrist_angle_deg':    data[3],
                'grip_angle_deg':     data[4],
                'flag':         data[5],
                'force_N':      data[6]
            }

            with self.lock:
                # 同期フラグが立ったら過去のゴミデータを一掃する
                if self.clear_flag:
                    self.sensor_logs = []
                    self.clear_flag = False
                    self.n_dropped = 0
                    self.n_dropped_sync = 0
                    self.n_after_clear = 0
                self.n_after_clear += 1
                if self.n_after_clear <= SYNC_SKIP_PACKETS:
                    self.n_dropped_sync += self.n_dropped
                    self.n_dropped = 0
                self.sensor_logs.append(sample)
        except:
            pass

    def clear_buffer_for_sync(self):
        """ GOの瞬間にUSBバッファとログリストを完全に空にして時刻0を合わせる """
        self.ser.reset_input_buffer()
        with self.lock:
            self.clear_flag = True

    def get_all_logs(self):
        with self.lock:
            return self.sensor_logs[:]

    def stop(self):
        self.running = False

class ExperimentController:
    def __init__(self, csv_name, user, venue):
        self.user, self.venue = user, venue
        self.csv_path = self._resolve_path(csv_name)
        print(f"[Init] Loading Sequence: {self.csv_path}")
        self.cmd_df = pd.read_csv(self.csv_path)
        gp = os.path.splitext(self.csv_path)[0] + "_guide.csv"
        self.guide = pd.read_csv(gp).sort_values("time").values.tolist() if os.path.exists(gp) else []
        if self.guide:
            print(f"[Init] 操作の指示: {gp}（{len(self.guide)} 件）")

        print(f"[Init] 圧力の枠: 1 つめ {SLOTS[0]}、2 つめ {SLOTS[1]}、3 つめ {SLOTS[2]}（config/pressure_slots.yaml）")
        print(f"[Init] Opening Serial Port {SERIAL_PORT} @ {BAUD_RATE}...")
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.receiver = SensorReceiver(self.ser)
            self.cmd_logs = [] # 指令値専用の履歴
        except serial.SerialException as e:
            print(f"[Error] Serial Port Open Failed: {e}")
            sys.exit(1)

    def _resolve_path(self, name):
        path = signal_path(name)
        if os.path.exists(path): return path
        if os.path.exists(name): return os.path.abspath(name)
        sys.exit(f"[Error] 入力信号が見つかりません: {path}")

    def run(self):
        self.receiver.start()
        print(f"\n=== STARTING ASYNC CONTROL ===")

        # 姿勢を初期化して安定するまで待機
        self._send([0,0,0])
        time.sleep(2.0)

        print("GO! (Synchronizing Clocks...)")
        # ここで受信バッファを捨て、次に受け取るデータを t = 0 にする
        self.receiver.clear_buffer_for_sync()

        try:
            t0_abs = time.perf_counter()
            self.t_go = t0_abs
            gi = 0
            for idx, row in self.cmd_df.iterrows():
                loop_start = time.perf_counter()
                while gi < len(self.guide) and self.guide[gi][0] <= idx * CONTROL_DT + 1e-9:   # 操作の指示（その時刻に表示）
                    print(f"[{self.guide[gi][0]:6.1f} s] {self.guide[gi][1]}", flush=True)
                    gi += 1

                cmd = [row['cmd_pressure_DF'], row['cmd_pressure_F'], row['cmd_pressure_G']]
                self._send(cmd)

                # 指令の理想の送信時刻（0.02 s 刻み）と、実際に送った時刻を記録する
                self.cmd_logs.append({
                    'cmd_time': idx * CONTROL_DT,
                    'cmd_DF': cmd[0],
                    'cmd_F': cmd[1],
                    'cmd_G': cmd[2],
                    'cmd_pc': time.perf_counter(),
                })

                # 50Hz周期維持
                while time.perf_counter() < t0_abs + (idx + 1) * CONTROL_DT:
                    time.sleep(0.0005)

        except KeyboardInterrupt:
            print("\nAborted.")
        finally:
            self.shutdown()

    def _send(self, cmd):
        cmd = [max(0.0, min(1.0, v)) for v in cmd]
        self.ser.write(HEADER + struct.pack(SEND_FMT, *cmd))

    def shutdown(self):
        self.elapsed = time.perf_counter() - getattr(self, "t_go", time.perf_counter())
        print("Shutting down...")
        self._send([0,0,0])
        self.receiver.stop()
        self.ser.close()
        self.save_logs()

    def save_logs(self):
        sensor_data = self.receiver.get_all_logs()
        if not sensor_data:
            print("[Error] No sensor data received.")
            return

        # 1. センサーデータ (200Hz) の再構築
        df_sensor = pd.DataFrame(sensor_data)
        # USBのブレを無視し、MicroLabBoxの完璧な200Hz(0.005s)クロックを信用して時刻を生成
        df_sensor['time'] = np.arange(len(df_sensor)) * 0.005

        # 2. 指令値データ (50Hz)
        df_cmd = pd.DataFrame(self.cmd_logs)
        df_cmd = df_cmd.rename(columns={'cmd_time': 'time'})

        # 3. センサーの各時刻(time)に対し、その瞬間にアクティブだった直近の指令値を合成 (Zero-Order Hold)
        df_merged = pd.merge_asof(
            df_sensor,
            df_cmd,
            on='time',
            direction='backward'
        )

        name = f"data_{os.path.splitext(os.path.basename(self.csv_path))[0]}_{int(time.time())}.csv"
        out_dir = run_dir(self.user, self.venue, "playback")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        df_merged.to_csv(path, index=False)
        print(f"\n[Saved] Reconstructed High-Res Log (200Hz) saved to:\n  -> {path}")
        # 受信の統計を json に（deploy_policy と同じ項目名。core_qc.py が読む）。書けなくてもログは残す
        try:
            expected = int(round(getattr(self, "elapsed", 0.0) * 200))
            meta = {"kind": "playback", "signal": os.path.basename(self.csv_path), "elapsed_s": round(getattr(self, "elapsed", 0.0), 3),
                    "packets_received": len(df_sensor), "packets_expected_at_200hz": expected,
                    "packet_yield": round(len(df_sensor) / expected, 4) if expected else None,
                    "bytes_discarded_on_resync": int(self.receiver.n_dropped),
                    "bytes_discarded_at_sync": int(self.receiver.n_dropped_sync)}
            with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=1)
            y = meta["packet_yield"]
            ytxt = " ({:.1f}%)".format(100 * y) if y else ""
            print("[Stat] 受信 {} / 期待 {}{}, 再同期で捨てたバイト {}".format(
                meta["packets_received"], expected, ytxt, meta["bytes_discarded_on_resync"]))
        except Exception as e:  # noqa: BLE001
            print(f"[Warn] 受信の統計の json を書けなかった：{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_name", type=str, help="signals/ の入力信号名（.csv は省略可）")
    add_user_venue_args(parser)
    args = parser.parse_args()
    user, venue = require_user_venue(args.user, args.venue)
    ExperimentController(args.csv_name, user, venue).run()