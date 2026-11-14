"""repo_paths.py — 実験データの置き場所を1か所で決める。

置き場所のルール（README「ディレクトリ構成」と同じ）:
  signals/                          ポリシー無しで再生する入力信号CSV（exp*, echo_*, pm_*, tm_*）
  data/<user>/<venue>/<kind>_<YYYYMMDD>/   実機で取ったデータ。1セッション1フォルダ
      <user>  : user0, user1, ...（porcaro_2026 の user と同じ番号）
      <venue> : 学会・目的ごとのフォルダ（access2026, okui2026, ...）
      <kind>  : gen2b など manifest のグループ名（deploy_policy.py）, playback（run_signal_playback.py）,
                verification（verify_sensors.py）,
                isotonic / isometric / rigcheck（tools/pamtest/ の PAM 単体試験）

user と venue は、各スクリプトの --user / --venue か、環境変数 PORCARO_USER / PORCARO_VENUE で指定する。
どちらも無ければ実行前に止める（data/ 直下やルートにまた散らからないように）。

    export PORCARO_USER=user0
    export PORCARO_VENUE=jfps2026
"""
from __future__ import annotations

import glob
import os
import re
import sys
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNALS_DIR = os.path.join(REPO_ROOT, "signals")
DATA_DIR = os.path.join(REPO_ROOT, "data")

_USER_RE = re.compile(r"^user\d+$")
_VENUE_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


# 出力先（実機データを書くスクリプト用）
def add_user_venue_args(parser) -> None:
    parser.add_argument("--user", type=str, default=None,
                        help="データの持ち主（user0, user1, ...）。省略時は環境変数 PORCARO_USER")
    parser.add_argument("--venue", type=str, default=None,
                        help="学会・目的のフォルダ名（例: jfps2026）。省略時は環境変数 PORCARO_VENUE")


def require_user_venue(user: str | None, venue: str | None) -> tuple[str, str]:
    """user / venue を引数か環境変数から決める。決まらなければ、実験を始める前に終了する。"""
    user = user or os.environ.get("PORCARO_USER")
    venue = venue or os.environ.get("PORCARO_VENUE")
    msg = []
    if not user:
        msg.append("--user（または環境変数 PORCARO_USER）が未指定です。例: --user user0")
    elif not _USER_RE.match(user):
        msg.append(f"user は user0, user1, ... の形で指定してください: {user}")
    if not venue:
        msg.append("--venue（または環境変数 PORCARO_VENUE）が未指定です。例: --venue jfps2026")
    elif not _VENUE_RE.match(venue):
        msg.append(f"venue は英小文字・数字・_ で指定してください: {venue}")
    if msg:
        sys.exit("[repo_paths] データの保存先が決まりません。\n  " + "\n  ".join(msg)
                 + "\n  まとめて指定するなら: export PORCARO_USER=user0 PORCARO_VENUE=jfps2026")
    venue_dir = os.path.join(DATA_DIR, user, venue)
    if not os.path.isdir(venue_dir):
        print(f"[repo_paths] 注意: {os.path.relpath(venue_dir, REPO_ROOT)}/ はまだ無いので新しく作ります"
              "（綴りが既存のフォルダと合っているか確認してください）")
    return user, venue


def run_dir(user: str, venue: str, kind: str, day: date | None = None) -> str:
    """data/<user>/<venue>/<kind>_<YYYYMMDD>/ のパス（作成はしない）。"""
    day = day or date.today()
    return os.path.join(DATA_DIR, user, venue, f"{kind}_{day:%Y%m%d}")


# 読み込み（解析スクリプト用）
def signal_path(name: str, root: str = REPO_ROOT) -> str:
    """入力信号 signals/<name>.csv のパス。"""
    if not name.endswith(".csv"):
        name += ".csv"
    return os.path.join(root, "signals", name)


def glob_measured(pattern: str, root: str = REPO_ROOT) -> list[str]:
    """信号再生の実測CSVを data/*/*/playback_*/ から探す（pattern はファイル名のglob）。"""
    return sorted(glob.glob(os.path.join(root, "data", "*", "*", "playback_*", pattern)))


def find_measured(fname: str, root: str = REPO_ROOT) -> str:
    """信号再生の実測CSV（data_<信号名>_<unixtime>.csv）をファイル名で1つ探す。"""
    hits = glob_measured(fname, root)
    if not hits:
        raise FileNotFoundError(f"実測データ {fname} が data/<user>/<venue>/playback_*/ に見つかりません"
                                f"（root={root}）")
    if len(hits) > 1:
        raise FileNotFoundError(f"実測データ {fname} が複数あります: {hits}")
    return hits[0]
