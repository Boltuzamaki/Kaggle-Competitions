"""Original TabM-style rank-one multihead MLP; official competition data only."""
from pathlib import Path
import gc, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, FOLDS, HEADS = "addicted_label", "id", 20260803, 5, 12

def locate():
    for p in Path("/kaggle/input").rglob("train.csv"):
        try: cols = pd.read_csv(p, nrows=1).columns
        except Exception: continue
        if TARGET in cols and (p.parent / "test.csv").exists():
            return p, p.parent / "test.csv"
    raise FileNotFoundError("competition tables not found")

def feature_frame(df):
    raw = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    numeric = raw.select_dtypes(include=np.number).columns.tolist()
    x = raw[numeric].astype("float32")
    for c in numeric: x[c + "__na"] = x[c].isna().astype("float32")
    x["missing_count"] = raw.isna().sum(axis=1).astype("float32")
    x["missing_pattern"] = sum(raw[c].isna().astype("int32") * (2 ** i) for i, c in enumerate(raw.columns))
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    if all(c in raw for c in parts):
        x["component_sum"] = raw[parts].sum(axis=1, min_count=1)
        if "daily_screen_time_hours" in raw:
            x["screen_residual"] = raw["daily_screen_time_hours"] - x["component_sum"]
    for name, a, b, eps in [
        ("screen_sleep_ratio", "daily_screen_time_hours", "sleep_hours", .25),
        ("weekend_daily_ratio", "weekend_screen_time", "daily_screen_time_hours", .25),
        ("notification_open_ratio", "notifications_per_day", "app_opens_per_day", 1.),
    ]:
        if a in raw and b in raw: x[name] = raw[a] / (raw[b] + eps)
    return x.replace([np.inf, -np.inf], np.nan)

class RankOneDense(tf.keras.layers.Layer):
    def __init__(self, units, heads=HEADS, activation="swish"):
        super().__init__(); self.units=units; self.heads=heads; self.activation=tf.keras.activations.get(activation)
    def build(self, shape):
        d=int(shape[-1]); init=tf.keras.initializers.RandomNormal(stddev=.04)
        self.kernel=self.add_weight(shape=(d,self.units),initializer="he_normal",name="kernel")
        self.r=self.add_weight(shape=(self.heads,d),initializer=tf.keras.initializers.Ones(),name="r")
        self.s=self.add_weight(shape=(self.heads,self.units),initializer=tf.keras.initializers.Ones(),name="s")
        self.bias=self.add_weight(shape=(self.heads,self.units),initializer="zeros",name="bias")
        self.r.assign_add(init(self.r.shape)); self.s.assign_add(init(self.s.shape))
    def call(self, x):
        if len(x.shape)==2: x=tf.repeat(x[:,None,:],self.heads,axis=1)
        z=tf.einsum("bhd,du->bhu",x*self.r[None,:,:],self.kernel)
        return self.activation(z*self.s[None,:,:]+self.bias[None,:,:])

def make_model(d):
    inp=tf.keras.Input((d,)); x=tf.keras.layers.GaussianNoise(.015)(inp)
    x=RankOneDense(256)(x); x=tf.keras.layers.LayerNormalization()(x); x=tf.keras.layers.Dropout(.10)(x)
    skip=x; x=RankOneDense(256)(x); x=tf.keras.layers.LayerNormalization()(x); x=tf.keras.layers.Dropout(.10)(x); x=x+skip
    x=RankOneDense(128)(x); logits=tf.keras.layers.Dense(1)(x)
    logits=tf.keras.layers.Reshape((HEADS,))(logits)
    model=tf.keras.Model(inp,logits)
    model.compile(tf.keras.optimizers.AdamW(8e-4,weight_decay=1e-5),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return model

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
train_path, test_path=locate(); train=pd.read_csv(train_path); test=pd.read_csv(test_path); y=train[TARGET].to_numpy("float32")
all_x=pd.concat([feature_frame(train),feature_frame(test)],ignore_index=True)
med=all_x.iloc[:len(train)].median(); all_x=all_x.fillna(med).fillna(0)
qt=QuantileTransformer(n_quantiles=1024,output_distribution="normal",subsample=300000,random_state=SEED)
X=qt.fit_transform(all_x.iloc[:len(train)]).astype("float32"); XT=qt.transform(all_x.iloc[len(train):]).astype("float32")
splitter=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(train)); fold_id=np.zeros(len(train),"int8"); test_preds=[]; scores=[]
for fold,(fit_idx,val_idx) in enumerate(splitter.split(X,y),1):
    fold_id[val_idx]=fold; model=make_model(X.shape[1]); checkpoint=f"/kaggle/working/tabm_fold{fold}.weights.h5"
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,min_delta=2e-5,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint(checkpoint,monitor="val_loss",save_best_only=True,save_weights_only=True)]
    model.fit(X[fit_idx],np.repeat(y[fit_idx,None],HEADS,axis=1),validation_data=(X[val_idx],np.repeat(y[val_idx,None],HEADS,axis=1)),epochs=25,batch_size=4096,callbacks=callbacks,verbose=2)
    vp=tf.sigmoid(model.predict(X[val_idx],batch_size=8192,verbose=0)).numpy().mean(axis=1)
    tp=tf.sigmoid(model.predict(XT,batch_size=8192,verbose=0)).numpy().mean(axis=1)
    oof[val_idx]=vp; test_preds.append(tp); scores.append(roc_auc_score(y[val_idx],vp)); tf.keras.backend.clear_session(); gc.collect()
pd.DataFrame({ID:train[ID],"fold":fold_id,"y":y,"pred":oof}).to_csv("/kaggle/working/oof_tabm_rank1.csv",index=False)
pd.DataFrame({ID:test[ID],TARGET:np.mean(test_preds,axis=0)}).to_csv("/kaggle/working/test_tabm_rank1.csv",index=False)
metrics={"model":"original TabM-style rank-one 12-head MLP","official_data_only":True,"seed":SEED,"folds":FOLDS,"fold_auc":scores,"oof_auc":roc_auc_score(y,oof),"device":[d.name for d in tf.config.list_physical_devices("GPU")]}
Path("/kaggle/working/metrics.json").write_text(json.dumps(metrics,indent=2)); print(metrics)
