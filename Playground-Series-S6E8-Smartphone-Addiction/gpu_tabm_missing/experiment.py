"""Missingness-gated multihead residual MLP; official competition data only."""
from pathlib import Path
import gc, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, FOLDS, HEADS = "addicted_label", "id", 20260803, 5, 8

def locate():
    for p in Path("/kaggle/input").rglob("train.csv"):
        try: cols=pd.read_csv(p,nrows=1).columns
        except Exception: continue
        if TARGET in cols and (p.parent/"test.csv").exists(): return p,p.parent/"test.csv"
    raise FileNotFoundError("competition tables not found")

def features(df):
    raw=df.drop(columns=[ID,TARGET],errors="ignore"); x=raw.select_dtypes(include=np.number).astype("float32").copy()
    cols=list(raw.columns); miss=raw.isna()
    for c in cols:
        if c in x: x[c+"__na"]=miss[c].astype("float32")
    x["missing_count"]=miss.sum(axis=1).astype("float32"); x["is_complete"]=(~miss.any(axis=1)).astype("float32")
    x["missing_pattern"]=sum(miss[c].astype("int32")*(2**i) for i,c in enumerate(cols))
    parts=["social_media_hours","gaming_hours","work_study_hours"]
    if all(c in raw for c in parts):
        x["component_sum"]=raw[parts].sum(axis=1,min_count=1)
        if "daily_screen_time_hours" in raw:
            x["screen_residual"]=raw.daily_screen_time_hours-x.component_sum
            for c in parts: x[c+"__share"]=raw[c]/(raw.daily_screen_time_hours+.25)
    pairs=[("screen_sleep","daily_screen_time_hours","sleep_hours",.25),("weekend_daily","weekend_screen_time","daily_screen_time_hours",.25),("notif_open","notifications_per_day","app_opens_per_day",1.)]
    for n,a,b,e in pairs:
        if a in raw and b in raw: x[n+"__ratio"]=raw[a]/(raw[b]+e)
    return x.replace([np.inf,-np.inf],np.nan)

def model(d, complete_index):
    inp=tf.keras.Input((d,)); base=tf.keras.layers.GaussianNoise(.01)(inp)
    base=tf.keras.layers.Dense(384,activation="swish")(base); base=tf.keras.layers.LayerNormalization()(base); base=tf.keras.layers.Dropout(.12)(base)
    heads=[]
    for h in range(HEADS):
        z=tf.keras.layers.Dense(192,activation="swish",name=f"adapter_{h}")(base)
        z=tf.keras.layers.LayerNormalization()(z); z=tf.keras.layers.Dropout(.08+0.01*(h%3))(z)
        z=tf.keras.layers.Dense(96,activation="swish")(z)
        heads.append(tf.keras.layers.Dense(1)(z))
    logits=tf.keras.layers.Concatenate(axis=1)(heads)
    # A learned complete/incomplete gate mixes independent heads without labels.
    complete=tf.keras.layers.Lambda(lambda z: z[:,complete_index:complete_index+1],name="complete_feature")(inp)
    gate=tf.keras.layers.Dense(HEADS,activation="softmax",name="missingness_gate")(complete)
    log_gate=tf.keras.layers.Lambda(lambda z: .05*tf.math.log(z+1e-7),name="log_gate")(gate)
    gated=tf.keras.layers.Add()([logits,log_gate])
    m=tf.keras.Model(inp,gated); m.compile(tf.keras.optimizers.AdamW(7e-4,weight_decay=2e-5),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)); return m

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
trp,tep=locate(); tr=pd.read_csv(trp); te=pd.read_csv(tep); y=tr[TARGET].to_numpy("float32")
fx=pd.concat([features(tr),features(te)],ignore_index=True); complete_index=fx.columns.get_loc("is_complete")
fx=fx.fillna(fx.iloc[:len(tr)].median()).fillna(0)
qt=QuantileTransformer(n_quantiles=1024,output_distribution="normal",subsample=300000,random_state=SEED)
X=qt.fit_transform(fx.iloc[:len(tr)]).astype("float32"); XT=qt.transform(fx.iloc[len(tr):]).astype("float32")
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); fold_id=np.zeros(len(tr),"int8"); tests=[]; scores=[]; regime=[]
for fold,(a,b) in enumerate(skf.split(X,y),1):
    fold_id[b]=fold; m=model(X.shape[1],complete_index); ck=f"/kaggle/working/missing_fold{fold}.weights.h5"
    cb=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,min_delta=2e-5,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint(ck,monitor="val_loss",save_best_only=True,save_weights_only=True)]
    m.fit(X[a],np.repeat(y[a,None],HEADS,1),validation_data=(X[b],np.repeat(y[b,None],HEADS,1)),epochs=25,batch_size=4096,callbacks=cb,verbose=2)
    vp=tf.sigmoid(m.predict(X[b],batch_size=8192,verbose=0)).numpy().mean(1); tp=tf.sigmoid(m.predict(XT,batch_size=8192,verbose=0)).numpy().mean(1)
    oof[b]=vp; tests.append(tp); scores.append(roc_auc_score(y[b],vp))
    original_complete=tr.iloc[b].drop(columns=[ID,TARGET]).notna().all(1).to_numpy(); regime.append({"fold":fold,"complete_auc":roc_auc_score(y[b][original_complete],vp[original_complete]),"incomplete_auc":roc_auc_score(y[b][~original_complete],vp[~original_complete])})
    tf.keras.backend.clear_session(); gc.collect()
pd.DataFrame({ID:tr[ID],"fold":fold_id,"y":y,"pred":oof}).to_csv("/kaggle/working/oof_tabm_missing.csv",index=False)
pd.DataFrame({ID:te[ID],TARGET:np.mean(tests,0)}).to_csv("/kaggle/working/test_tabm_missing.csv",index=False)
metrics={"model":"original missingness-gated 8-head residual MLP","official_data_only":True,"seed":SEED,"folds":FOLDS,"fold_auc":scores,"oof_auc":roc_auc_score(y,oof),"regime_auc":regime,"device":[d.name for d in tf.config.list_physical_devices("GPU")]}
Path("/kaggle/working/metrics.json").write_text(json.dumps(metrics,indent=2)); print(metrics)
