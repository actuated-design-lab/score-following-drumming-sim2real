"""Read the research-side data package (data/package_sec4_20261109.md) into DataFrames.

The package is a Markdown document whose tables are script outputs of the research side.
This module only parses those tables; it does not compute any new value.
"""
import re
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
PACKAGE = DATA / "package_sec4_20261109.md"

# Internal names of the package -> names used in the paper.
METHOD = {"B": "LSTM", "E": "MLP", "F0": "FF", "F3": "FF-cal"}
TASK = {"holdout0": "Held-out 1", "holdout1": "Held-out 2", "gmd138": "GMD 138",
        "gmd170": "GMD 170", "double160": "Double 160"}
TASK_ORDER = ["Held-out 1", "Held-out 2", "GMD 138", "GMD 170", "Double 160"]
METHOD_ORDER = ["LSTM", "MLP", "FF", "FF-cal"]
BIN_ORDER = ["<120", "120-200", "200-320", ">=320"]


def _num(s):
    s = s.strip().replace("−", "-").replace("**", "")
    return float(s)


def _tables(text):
    """Return a list of (heading, rows) for every Markdown table, rows as lists of cells."""
    out, heading, rows = [], "", None
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if rows is None:
                rows = [cells]
            elif not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif rows is not None:
            out.append((heading, rows)); rows = None
    if rows is not None:
        out.append((heading, rows))
    return out


def _find(tables, heading_startswith, header0):
    for h, rows in tables:
        if h.startswith(heading_startswith) and rows[0][0] == header0:
            return rows
    raise KeyError((heading_startswith, header0))


def _bin(label):
    label = label.replace(" ms", "").replace("（最初を含む）", "").replace("–", "-").strip()
    return {"< 120": "<120", "≥ 320": ">=320"}.get(label, label)


def _task(label):
    return TASK[re.sub(r"（.*）", "", label).strip()]


def load():
    text = PACKAGE.read_text(encoding="utf-8")
    t = _tables(text)
    d = {}
    # 1.2 FF parameters
    rows = _find(t, "1.2", "区分（前打間隔）")
    d["ff_params"] = pd.DataFrame(
        [[_bin(r[0])] + [_num(x) for x in r[1:]] for r in rows[1:]],
        columns=["bin", "L_ms", "h_ms", "a_up_MPa", "a_dn_MPa", "g_up_MPa", "g_dn_MPa", "delta_ms"])
    # 2 main table per task
    rows = _find(t, "2.", "方法")
    rec = []
    for r in rows[1:]:
        if len(r) != 9:
            continue
        m, s = r[3].split("±")
        rec.append([METHOD[r[0]], _task(r[1]), int(r[2]), _num(m), _num(s), _num(r[4]),
                    _num(r[5]), _num(r[6]), _num(r[7]), _num(r[8])])
    d["hw_task"] = pd.DataFrame(rec, columns=["method", "task", "onsets", "success_mean", "success_sd",
                                              "success_old", "not_struck", "early", "late", "pushed"])
    # 2 five-task mean (second table under heading 2 whose first header is 方法 and has 7 columns)
    rows = [rs for h, rs in t if h.startswith("2.") and rs[0][0] == "方法" and len(rs[0]) == 7][0]
    rec = []
    for r in rows[1:]:
        m, s = r[1].split("±")
        rec.append([METHOD[r[0]], _num(m), _num(s), _num(r[2]), _num(r[3]), _num(r[4]), _num(r[5]), _num(r[6])])
    d["hw_mean5"] = pd.DataFrame(rec, columns=["method", "success_mean", "success_sd", "success_old",
                                               "not_struck", "early", "late", "pushed"])
    # 3 simulation nominal vs hardware
    rows = _find(t, "3.", "タスク")
    rec = []
    for r in rows[1:]:
        task = "Mean" if r[0].startswith("5 タスク") else _task(r[0])
        rec.append([task] + [_num(x) for x in r[1:]])
    d["sim_hw"] = pd.DataFrame(rec, columns=["task", "FF_sim", "FF_hw", "MLP_sim", "MLP_hw", "LSTM_sim", "LSTM_hw"])
    # 4 IOI bins
    rows = _find(t, "4.", "前打間隔 [ms]")
    rec = []
    for r in rows[1:]:
        for col, cell in zip(rows[0][1:], r[1:]):
            v, n = re.match(r"([\d.]+)（(\d+)）", cell).groups()
            rec.append([_bin(r[0]), METHOD[col], float(v), int(n)])
    d["ioi"] = pd.DataFrame(rec, columns=["bin", "method", "success", "onsets"])
    # 4 rest / within (two tables: all tasks, without double160)
    rs = [rows for h, rows in t if h.startswith("休符明けと連続の中")]
    rec = []
    for subset, rows in zip(["all", "no_double160"], rs):
        for r in rows[1:]:
            for col, cell in zip(rows[0][1:], r[1:]):
                v, n = re.match(r"([\d.]+)（(\d+)）", cell).groups()
                rec.append([subset, {"休符明け": "after_rest", "連続の中": "within_run"}[r[0]], METHOD[col], float(v), int(n)])
    d["rest"] = pd.DataFrame(rec, columns=["subset", "context", "method", "success", "onsets"])
    # 5 downswing per bin
    rows = _find(t, "5.", "区分（前打間隔）")
    d["swing"] = pd.DataFrame([[_bin(r[0])] + [_num(x) for x in r[1:]] for r in rows[1:]],
                              columns=["bin", "a_dn_MPa", "onsets", "omega_deg_s", "A_deg", "theta_top_deg",
                                       "PF_meas_MPa", "success"])
    # 5 cross table by previous bin
    rows = _find(t, "5.", "区分")
    rec = []
    for r in rows[1:]:
        for col, cell in zip(rows[0][1:], r[1:]):
            a, s, n = re.match(r"([\d.]+)° / ([\d.]+)（(\d+)）", cell).groups()
            rec.append([_bin(r[0]), _bin(col.replace("直前", "")), float(a), float(s), int(n)])
    d["swing_prev"] = pd.DataFrame(rec, columns=["bin", "prev_bin", "A_median_deg", "success", "onsets"])
    return d, text
