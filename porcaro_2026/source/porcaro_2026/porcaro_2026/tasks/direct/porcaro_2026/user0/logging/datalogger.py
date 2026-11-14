from __future__ import annotations
import csv
import os
from typing import Sequence
import atexit

class DataLogger:
    """行を貯めて CSV に追記する（LoggingManager から、環境 0 の行だけが来る）。終了時にも save する。"""
    def __init__(self, filepath: str, headers: Sequence[str]):
        self.filepath = filepath
        self.headers = list(headers)
        self.buffer: list[list[float]] = []

        # 既存のファイルは上書きする
        self._write_header()

        atexit.register(self.save)

    def _write_header(self):
        try:
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            print(f"[DataLogger] ヘッダーを {self.filepath} に書き込みました。")
        except OSError as e:
            print(f"[DataLogger] ERROR: ヘッダーの書き込みに失敗しました。 {e}")

    def add_step(self, data_row: Sequence[float]):
        """1 行を貯める。列数がヘッダーと違う行は捨てる。"""
        if len(data_row) != len(self.headers):
            print(f"[DataLogger] WARN: データ長({len(data_row)})がヘッダー長({len(self.headers)})と一致しません。スキップします。")
            return
        self.buffer.append(list(data_row))

    def add_data_batch(self, rows: list[list]):
        self.buffer.extend(rows)

    def save(self):
        """貯めた行を追記してバッファを空にする。"""
        if not self.buffer:
            print("[DataLogger] バッファが空です。書き込みをスキップしました。")
            return

        try:
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(self.buffer)

            num_rows = len(self.buffer)
            self.buffer.clear()
            print(f"[DataLogger] {num_rows} 件のデータを {self.filepath} に追記しました。")

        except OSError as e:
            print(f"[DataLogger] ERROR: CSVファイルへの書き込みに失敗しました。 {e}")

    def clear_buffer(self):
        self.buffer.clear()