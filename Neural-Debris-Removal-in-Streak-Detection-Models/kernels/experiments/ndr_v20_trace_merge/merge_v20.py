"""Merge two V20 TRACE shards into frozen suppression-only submissions."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/"output_local"; OUT.mkdir(parents=True,exist_ok=True)
EXPECTED="4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412"
ANCHOR=ROOT/"forensics/kaggle_ndr_v19_pcgrad_stability/output_v1/submission_V19_0_exact_v15b.csv"
SHARDS=[ROOT/f"forensics/kaggle_ndr_v20_trace_shard{i}/output_v2/ndr_v20_shard{i}/trace_diagnostics_shard{i}.csv" for i in range(2)]
VARIANTS={
 "V20_0_exact_v15b":{"mode":"identity"},
 "V20_A_strict":{"threshold":"strict_threshold","pcgrad_column":"pcgrad_votes70","votes":2,"floor":.02},
 "V20_B_relaxed":{"threshold":"relaxed_threshold","pcgrad_column":"pcgrad_votes65","votes":2,"floor":.02},
 "V20_C_trace_extreme":{"threshold":"strict_threshold","pcgrad_column":"pcgrad_votes55","votes":1,"floor":.02},
 "V20_D_four_signal":{"threshold":"relaxed_threshold","pcgrad_column":"pcgrad_votes60","votes":3,"cap":.10,"floor":.02},
}

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for c in iter(lambda:f.read(1<<20),b""):h.update(c)
 return h.hexdigest()
def parse(v):
 t=str(v).strip()
 if not t or t=="nan":return np.zeros((0,5),np.float32)
 a=np.asarray(list(map(float,t.split())),np.float32);return a.reshape(-1,5)
def fmt(a):
 return " ".join(f"{float(x):.6f}" if j%5==0 else f"{float(x):.2f}" for j,x in enumerate(a.ravel())) if len(a) else " "

assert sha(ANCHOR)==EXPECTED
anchor=pd.read_csv(ANCHOR,dtype={"image_id":str}); parts=[pd.read_csv(p,dtype={"image_id":str}) for p in SHARDS]; diag=pd.concat(parts,ignore_index=True)
gates=[json.loads((p.parent/"final_report.json").read_text(encoding="utf-8"))["gate"] for p in SHARDS]
assert all(g["enabled"]==gates[0]["enabled"] for g in gates); gate=gates[0]
lookup={(str(r.image_id),int(r.candidate)):r for r in diag.itertuples(index=False)}
report={}
for name,spec in VARIANTS.items():
 frame=anchor.copy(); rendered=[]; changed=0; mass=0.0
 for row in anchor.itertuples(index=False):
  arr=parse(row.prediction_string); scores=arr[:,0].copy()
  if name!="V20_0_exact_v15b" and gate["enabled"]:
   for i in range(len(arr)):
    key=(str(row.image_id),i)
    if scores[i]<.21-1e-6 or key not in lookup:continue
    d=lookup[key]; threshold=float(gate[spec["threshold"]])
    if float(d.trace_probability)>=threshold and int(getattr(d,spec["pcgrad_column"]))>=spec["votes"]:
     new=min(float(scores[i]),spec.get("cap",spec["floor"]));
     if name=="V20_D_four_signal" and float(d.trace_probability)>=float(gate["strict_threshold"]):new=min(new,spec["floor"])
     mass+=float(scores[i]-new);changed+=int(new<scores[i]-1e-7);scores[i]=new
  arr[:,0]=scores;rendered.append(fmt(arr))
 frame["prediction_string"]=rendered; path=OUT/f"submission_{name}.csv"
 if name=="V20_0_exact_v15b":path.write_bytes(ANCHOR.read_bytes())
 else:frame.to_csv(path,index=False)
 report[name]={"sha256":sha(path),"changed":changed,"removed_mass":mass,"rows":len(frame),"unique_ids":int(frame.image_id.nunique()),"boxes":int(sum(len(parse(v)) for v in frame.prediction_string)),"boxes_added":0,"boxes_moved":0,"confidence_increases":0}
assert report["V20_0_exact_v15b"]["sha256"]==EXPECTED
(OUT/"final_report.json").write_text(json.dumps({"status":"complete","gate":gate,"variants":report,"rule_7a_guard_passed":True,"competition_submission_created":False},indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
