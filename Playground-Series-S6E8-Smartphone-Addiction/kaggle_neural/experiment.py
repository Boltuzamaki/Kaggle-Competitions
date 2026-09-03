"""Original GPU neural tabular ensemble using competition tables only."""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, FOLDS = "addicted_label", "id", 20260803, 5

def locate():
    for p in Path('/kaggle/input').rglob('train.csv'):
        try: cols=pd.read_csv(p,nrows=1).columns
        except: continue
        if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv',p.parent/'sample_submission.csv'
    raise FileNotFoundError('competition tables not found')

def features(d):
    x=d.drop(columns=[ID,TARGET],errors='ignore').copy()
    base=list(x.columns)
    for c in base: x[c+'__missing']=x[c].isna().astype('float32')
    x['missing_count']=d[base].isna().sum(1)
    def ratio(n,a,b,k=.25):
        if a in d and b in d: x[n]=d[a]/(d[b]+k)
    parts=['social_media_hours','gaming_hours','work_study_hours']
    if all(c in d for c in parts):
        x['component_sum']=d[parts].sum(1,min_count=1)
        if 'daily_screen_time_hours' in d: x['screen_residual']=d.daily_screen_time_hours-x.component_sum
    ratio('weekend_ratio','weekend_screen_time','daily_screen_time_hours')
    ratio('screen_sleep_ratio','daily_screen_time_hours','sleep_hours')
    ratio('notif_open_ratio','notifications_per_day','app_opens_per_day',1)
    for c in x.select_dtypes(exclude=np.number): x[c]=pd.factorize(x[c].astype('string'),sort=True)[0]
    return x.astype('float32')

def make_net(n):
    inp=tf.keras.Input((n,)); x=tf.keras.layers.Dense(512)(inp); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.Activation('swish')(x); x=tf.keras.layers.Dropout(.18)(x)
    x=tf.keras.layers.Dense(256)(x); x=tf.keras.layers.BatchNormalization()(x); x=tf.keras.layers.Activation('swish')(x); x=tf.keras.layers.Dropout(.12)(x)
    x=tf.keras.layers.Dense(128,activation='swish')(x); out=tf.keras.layers.Dense(1,activation='sigmoid')(x)
    m=tf.keras.Model(inp,out); m.compile(tf.keras.optimizers.AdamW(1.5e-3,weight_decay=2e-4),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='auc')]); return m

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
tp,sp,subp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); sample=pd.read_csv(subp); y=tr[TARGET].values.astype('float32')
allx=pd.concat([features(tr),features(te)],ignore_index=True).replace([np.inf,-np.inf],np.nan)
allx=allx.fillna(allx.iloc[:len(tr)].median())
qt=QuantileTransformer(n_quantiles=1000,output_distribution='normal',subsample=200000,random_state=SEED)
X=qt.fit_transform(allx.iloc[:len(tr)]).astype('float32'); Xt=qt.transform(allx.iloc[len(tr):]).astype('float32')
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); folds=np.zeros(len(tr),dtype='int8'); tests=[]; scores=[]
for f,(a,b) in enumerate(skf.split(X,y),1):
    folds[b]=f; model=make_net(X.shape[1]); ck=f'/kaggle/working/fold{f}.weights.h5'
    cb=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',mode='max',patience=5,min_delta=1e-5),tf.keras.callbacks.ModelCheckpoint(ck,monitor='val_auc',mode='max',save_best_only=True,save_weights_only=True)]
    model.fit(X[a],y[a],validation_data=(X[b],y[b]),epochs=30,batch_size=4096,callbacks=cb,verbose=2)
    model.load_weights(ck); oof[b]=model.predict(X[b],batch_size=8192,verbose=0).ravel(); tests.append(model.predict(Xt,batch_size=8192,verbose=0).ravel()); scores.append(roc_auc_score(y[b],oof[b])); tf.keras.backend.clear_session()
pd.DataFrame({ID:tr[ID],'fold':folds,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_neural.csv',index=False)
sample[TARGET]=np.mean(tests,0); sample.to_csv('/kaggle/working/test_neural.csv',index=False)
metrics={'provenance':'competition data only','fold_auc':scores,'oof_auc':roc_auc_score(y,oof),'device':[x.name for x in tf.config.list_physical_devices('GPU')]}
Path('/kaggle/working/metrics.json').write_text(json.dumps(metrics,indent=2)); print(metrics)
