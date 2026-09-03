#!/usr/bin/env python3
"""Poll, ingest, validate, and stack explicitly allowed private S6E8 kernels.

Never submits. Never discovers or consumes arbitrary Kaggle/public predictions.
Set KAGGLE_COMMAND when IPv4 wrapping is required, for example:
KAGGLE_COMMAND='/path/to/kaggle-python kaggle_ipv4.py' .venv/bin/python ...
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, shlex, shutil, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; TARGET="addicted_label"; SEED=20260803
REGISTRY=HERE/"kernel_registry.json"; ARTIFACTS=ROOT/"artifacts"/"automated_original"
REPORTS=ROOT/"ensemble_original"/"reports"; NOTES=ROOT/"NOTES.md"

def run_kaggle(args):
    command=shlex.split(os.environ.get("KAGGLE_COMMAND","kaggle"))+args
    return subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=120)

def status(slug):
    p=run_kaggle(["kernels","status",slug]); text=(p.stdout+p.stderr).strip()
    if p.returncode: return "ERROR",text
    for state in ("COMPLETE","RUNNING","ERROR","CANCELLED","QUEUED"):
        if state in text: return state,text
    return "UNKNOWN",text

def fingerprint(ids): return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()[:16]

def validate_source(item):
    source=ROOT/item["source"]
    if not source.is_file(): raise ValueError(f"owned source missing: {source}")
    body=source.read_text(errors="replace").lower()
    if "public_outputs" in body or "research_notebooks" in body:
        raise ValueError("source references prohibited prediction directories")
    if "competitions submit" in body or "api.competition_submit" in body:
        raise ValueError("source contains submission API")

def find_file(root,name):
    hits=list(root.rglob(name))
    if len(hits)!=1: raise ValueError(f"expected exactly one {name}, found {len(hits)}")
    return hits[0]

def align(frame,reference,kind):
    if "id" not in frame or len(frame)!=len(reference) or not frame.id.is_unique or set(frame.id)!=set(reference.id):
        raise ValueError(f"{kind} ID set/count invalid")
    return frame.set_index("id").reindex(reference.id).reset_index()

def values(frame,spec):
    if isinstance(spec,str):
        if spec not in frame: raise ValueError(f"missing prediction column {spec}")
        out=frame[spec].to_numpy(float)
    else:
        left,right,w=spec["blend"]
        if left not in frame or right not in frame: raise ValueError("blend source columns absent")
        out=(1-float(w))*frame[left].to_numpy(float)+float(w)*frame[right].to_numpy(float)
    if not np.isfinite(out).all() or np.min(out)<0 or np.max(out)>1: raise ValueError("non-finite/out-of-range predictions")
    return out

def ingest(item,download,train,test):
    validate_source(item); o=align(pd.read_csv(find_file(download,item["oof"])),train,"OOF")
    t=align(pd.read_csv(find_file(download,item["test"])),test,"test")
    label="y" if "y" in o else TARGET if TARGET in o else None
    if label is None or not np.array_equal(o[label].to_numpy(int),train[TARGET].to_numpy(int)): raise ValueError("OOF labels absent/misaligned")
    if "fold" not in o or o.fold.isna().any() or o.fold.nunique()<3 or (o.fold<0).any(): raise ValueError("full OOF fold assignment absent")
    dest=ARTIFACTS/item["key"]; dest.mkdir(parents=True,exist_ok=True); registered=[]
    for stream,(oc,tc) in item["streams"].items():
        op,tp=values(o,oc),values(t,tc); name=f'{item["key"]}__{stream}'
        pd.DataFrame({"id":train.id,"fold":o.fold.astype(int),"y":train[TARGET],"pred":op}).to_csv(dest/f"oof__{stream}.csv",index=False)
        pd.DataFrame({"id":test.id,"pred":tp}).to_csv(dest/f"test__{stream}.csv",index=False)
        registered.append({"name":name,"kernel":item["slug"],"oof":str((dest/f"oof__{stream}.csv").relative_to(ROOT)),"test":str((dest/f"test__{stream}.csv").relative_to(ROOT)),"auc":roc_auc_score(train[TARGET],op)})
    return registered

def load_audit():
    p=ROOT/"ensemble_original"/"audit_and_blend.py"; s=importlib.util.spec_from_file_location("ab",p)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def crossfit(x,y,ab):
    pred=np.zeros(len(y)); cv=StratifiedKFold(5,shuffle=True,random_state=SEED+91)
    for fit,val in cv.split(x,y):
        w=ab.fit_weights(x[fit],y[fit],"logit"); pred[val]=ab.combine(x[val],w,"logit")
    return roc_auc_score(y,pred)

def guarded_stack(records,train,test,threshold):
    ab=load_audit()
    # Immutable benchmark: later registrations in audit_and_blend.py must not
    # silently move the comparison base.
    core_names=["xgb_te_5fold","xgb_te_4fold","catboost_unique","histgb_5fold",
                "neural_5fold","raw_xgb_bag","lgb_te_5fold","tabm_rank1_v2"]
    oofs=[]; tests=[]; names=[]
    for name in core_names:
        op,tp,_=ab.load_pair(name,ab.MODELS[name],train,test); names.append(name); oofs.append(op); tests.append(tp)
    core_count=len(names)
    for r in records:
        o=align(pd.read_csv(ROOT/r["oof"]),train,"registered OOF"); t=align(pd.read_csv(ROOT/r["test"]),test,"registered test")
        names.append(r["name"]); oofs.append(o.pred.to_numpy(float)); tests.append(t.pred.to_numpy(float))
    x,xt=np.column_stack(oofs),np.column_stack(tests); y=train[TARGET].to_numpy(int)
    core_auc=crossfit(x[:,:core_count],y,ab); rows=[{"configuration":"core","auc":core_auc}]
    for j in range(core_count,len(names)): rows.append({"configuration":"core_plus_"+names[j],"auc":crossfit(x[:,list(range(core_count))+[j]],y,ab)})
    if len(names)>core_count: rows.append({"configuration":"core_plus_all_new","auc":crossfit(x,y,ab)})
    report=pd.DataFrame(rows).sort_values("auc",ascending=False); report.to_csv(REPORTS/"automated_candidate_crossfit.csv",index=False)
    best=report.iloc[0]; gain=float(best.auc-core_auc); candidate=REPORTS/"candidate_submission.csv"
    created=False
    if gain>=threshold and best.configuration!="core":
        keep=(list(range(len(names))) if best.configuration=="core_plus_all_new"
              else list(range(core_count)))
        if best.configuration not in ("core_plus_all_new","core"):
            selected=best.configuration.removeprefix("core_plus_"); keep += [names.index(selected)]
        w=ab.fit_weights(x[:,keep],y,"logit"); pred=ab.combine(xt[:,keep],w,"logit")
        pd.DataFrame({"id":test.id,TARGET:pred}).to_csv(candidate,index=False); created=True
    elif candidate.exists(): candidate.unlink()
    return core_auc,str(best.configuration),gain,created

def append_notes(statuses,records,result):
    marker="### Automated original-kernel ingestion status"
    block=["",marker,"",f"- Updated `{time.strftime('%Y-%m-%d %H:%M:%S')}` by `automation/monitor_original_kernels.py`."]
    block += [f"- `{key}`: **{state}**." for key,state in statuses.items()]
    block += [f"- Registered `{r['name']}`: OOF AUC `{r['auc']:.8f}`." for r in records]
    if result:
        core,best,gain,created=result; block += [f"- Guarded core AUC `{core:.8f}`; best configuration `{best}`; gain `{gain:+.8f}`.",f"- Candidate submission created: **{created}**. Threshold is `+0.00030`; nothing is submitted automatically."]
    text=NOTES.read_text(); start=text.find(marker)
    if start>=0: text=text[:start].rstrip()+"\n"
    NOTES.write_text(text+"\n".join(block)+"\n")

def once(config):
    train,test=pd.read_csv(ROOT/"train.csv"),pd.read_csv(ROOT/"test.csv")
    ARTIFACTS.mkdir(parents=True,exist_ok=True); statuses={}; records=[]
    registry_path=ARTIFACTS/"registry.json"
    prior=json.loads(registry_path.read_text()) if registry_path.exists() else []
    by_name={r["name"]:r for r in prior}
    for item in config["kernels"]:
        if not item.get("enabled",True): statuses[item["key"]]="DISABLED_PENDING_SLUG"; continue
        state,detail=status(item["slug"]); statuses[item["key"]]=state
        if state!="COMPLETE": continue
        with tempfile.TemporaryDirectory(prefix="s6e8_kernel_") as td:
            p=run_kaggle(["kernels","output",item["slug"],"-p",td])
            if p.returncode: statuses[item["key"]]="DOWNLOAD_ERROR"; continue
            try:
                for r in ingest(item,Path(td),train,test): by_name[r["name"]]=r
            except Exception as exc: statuses[item["key"]]="VALIDATION_ERROR: "+str(exc)
    records=sorted(by_name.values(),key=lambda r:r["name"]); registry_path.write_text(json.dumps(records,indent=2)+"\n")
    result=guarded_stack(records,train,test,float(config["robust_gain_threshold"])) if records else None
    append_notes(statuses,records,result); print(json.dumps({"statuses":statuses,"registered":len(records),"stack":result},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--watch",action="store_true"); ap.add_argument("--interval",type=int,default=60); args=ap.parse_args()
    config=json.loads(REGISTRY.read_text())
    while True:
        once(config)
        if not args.watch: break
        active=[k for k in config["kernels"] if k.get("enabled",True)]
        if all(status(k["slug"])[0] in {"COMPLETE","ERROR","CANCELLED"} for k in active): break
        time.sleep(max(30,args.interval))
if __name__=="__main__": main()
