"""c5_train_rule.py — 押し付けの判定の案（11/04）：H 未満の離れでつながった接触の連なりが 100 ms を超えたら、中の接触をまとめて打撃から外す（CPU）

短い連なり（100 ms 以内）は手を付けない（中の打撃は元のまま別々に採点する）。H = 0 は今の採点（contact_event）
実機：本表 60 本（core_score と同じ時刻・同じ extract_strikes_real）。sim：run_1104.sh trace の DR なし（nom_*）
出力：data/user0/eval/access_1104/c5/{c5_train_rule.csv, c5_train_rule.md}

  python scripts/tools/user0/access/c5_train_rule.py
"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0,"scripts/tools/user0/access"); sys.path.insert(0,"scripts/tools/user0/ff")
import realcore as R, ff_score
from c3_sim_real import SIM
cs=R._core_score()
from midi_rhythm_generator import MidiRhythmGenerator
from strike_metrics import extract_strikes_real
def train_mask(t,f,H,on=1.0,maxlen=0.100):
    """1 N 以上の区間を、離れが H 秒未満ならつないだ「接触の連なり」にし、長さが maxlen を超える連なりの中の力を 0 にする（打撃に数えない）。
    短い連なりは手を付けない（中の打撃は元のまま別々に数える）"""
    f=f.copy(); above=f>=on; n=len(f)
    segs=[]; i=0
    while i<n:
        if above[i]:
            j=i
            while j<n and above[j]: j+=1
            segs.append([i,j]); i=j
        else: i+=1
    if not segs: return f, 0
    tr=[segs[0][:]]
    for s,e in segs[1:]:
        if t[s]-t[tr[-1][1]-1]<H: tr[-1][1]=e
        else: tr.append([s,e])
    killed=0
    for s,e in tr:
        if t[e-1]-t[s]>maxlen: f[s:e]=0.0; killed+=1
    return f, killed
def real_score(file,kind,stem,H):
    p,_=cs.orig_midi(stem)
    g=MidiRhythmGenerator(p,device=torch.device("cpu"),dt=0.02,target_force=20.0,lookahead_steps=1)
    buf=g.trajectory_buffer.cpu().numpy(); buf=buf*20 if buf.max()<=1.5 else buf
    df=R.load(file,kind); t=df.time.to_numpy(float)
    k=np.clip((t/0.02).astype(int),0,len(buf)-1); df["target_force"]=np.where(t/0.02<len(buf),buf[k],0.0)
    if H>0: df["force_N"]=train_mask(t,df.force_N.to_numpy(float),H)[0]
    return extract_strikes_real(df,bpm=float(g.bpm),target_ref=20.0,force_threshold=1.0,method="contact_event").success.mean()
def sim_score(m,task,H):
    z=np.load(SIM/f"nom_{m}_{task}/trace.npz"); tt=z["time"]; f=z["force"].copy()
    if H>0: f=train_mask(tt,f,H)[0]
    ts=pd.DataFrame({"time_s":tt,"target_force":z["target"][:len(tt)],"force_z":f}); ts=ts[tt<=z["onsets"][-1]+1.3]
    return ff_score.score_sim(ts,float(z["bpm"])).success.mean()
rr=R.runs(); rows=[]
for (m,task),g in rr.groupby(["method","task"]):
    for H in [0,0.02,0.03,0.04,0.06]:
        rows.append(dict(src="real",m=m,task=task,H=H,s=np.mean([real_score(f,k,task,H) for f,k in zip(g.file,g.kind_log)])))
        rows.append(dict(src="sim_nom",m=m,task=task,H=H,s=sim_score(m,task,H)))
D=pd.DataFrame(rows)
OUT=R.ROOT/"data/user0/eval/access_1104/c5"; OUT.mkdir(parents=True, exist_ok=True); D.to_csv(OUT/"c5_train_rule.csv",index=False)
T=D.groupby(["src","m","H"]).s.mean().unstack("H"); T.columns=[f"H{int(h*1000)}ms" for h in T.columns]
print(T.round(3).to_string())
P=D[(D.H==0.04)].pivot_table(index=["m","task"],columns="src",values="s")
print(P.round(3).to_string())
open(OUT/"c5_train_rule.md","w").write("# 押し付けの判定の案：連なりが 100 ms を超えたら外す（5 タスクの平均、sim は DR なし）\n\n出典：scripts/tools/user0/access/c5_train_rule.py\n\n"
    + T.round(3).to_markdown() + "\n\n## H = 40 ms のタスクごと\n\n" + P.round(3).to_markdown() + "\n")
