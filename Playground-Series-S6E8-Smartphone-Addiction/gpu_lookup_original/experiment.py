"""Fold-safe TensorFlow lookup transformer; official competition data only."""
from pathlib import Path
import gc,json,random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
TARGET,ID,SEED,FOLDS='addicted_label','id',20260803,5
def locate():
    for p in Path('/kaggle/input').rglob('train.csv'):
        try: cols=pd.read_csv(p,nrows=1).columns
        except Exception: continue
        if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
    raise FileNotFoundError('official competition tables not found')
def derived(df):
    c=['social_media_hours','gaming_hours','work_study_hours']; s=df[c].sum(axis=1,min_count=3)
    return pd.DataFrame({'other_screen':df.daily_screen_time_hours-s,'component_sum':s,'other_frac':(df.daily_screen_time_hours-s)/(df.daily_screen_time_hours+.1),'weekend_gap':df.weekend_screen_time-df.daily_screen_time_hours,'weekend_other':df.weekend_screen_time-s,'component_frac':s/(df.daily_screen_time_hours+.1)})
def k(s): return s.astype('string').fillna('__NA__')
def lookup_arrays(fit,*others):
    arrays=[np.zeros((len(z),len(fit.columns)),dtype='int32') for z in (fit,)+others]; offsets=[]; total=0
    for j,c in enumerate(fit.columns):
        vals=pd.Index(k(fit[c]).unique()); mp={v:i+1 for i,v in enumerate(vals)}; offsets.append(total)
        for arr,z in zip(arrays,(fit,)+others): arr[:,j]=k(z[c]).map(mp).fillna(0).to_numpy('int32')+total
        total+=len(vals)+1
    return arrays,total
def quantile_arrays(fit,*others):
    outs=[np.zeros((len(z),fit.shape[1]),dtype='float32') for z in (fit,)+others]; masks=[]
    for z in (fit,)+others: masks.append(z.isna().to_numpy('float32'))
    for j,c in enumerate(fit.columns):
        # Exact-value lookup embeddings handle categorical columns.  PLR is a
        # continuous-value representation, so categorical PLR inputs stay zero
        # instead of imposing an arbitrary ordinal conversion.
        if not pd.api.types.is_numeric_dtype(fit[c]):
            continue
        ok=fit[c].notna(); qt=QuantileTransformer(n_quantiles=min(1000,max(10,int(ok.sum()))),output_distribution='normal',subsample=250000,random_state=SEED+j)
        qt.fit(fit.loc[ok,[c]])
        for out,z in zip(outs,(fit,)+others):
            good=z[c].notna(); out[good.to_numpy(),j]=qt.transform(z.loc[good,[c]]).ravel().astype('float32')
    return outs,masks
class PLR(tf.keras.layers.Layer):
    def __init__(self,nfeat,kfreq,d): super().__init__(); self.nfeat=nfeat; self.kfreq=kfreq; self.d=d
    def build(self,shape):
        self.freq=self.add_weight(shape=(self.nfeat,self.kfreq),initializer=tf.keras.initializers.RandomNormal(stddev=.5)); self.proj=self.add_weight(shape=(self.nfeat,2*self.kfreq,self.d),initializer='glorot_uniform'); self.bias=self.add_weight(shape=(self.nfeat,self.d),initializer='zeros')
    def call(self,x):
        z=2*np.pi*x[:,:,None]*self.freq[None,:,:]; z=tf.concat([tf.sin(z),tf.cos(z)],axis=-1); return tf.einsum('bfk,fkd->bfd',z,self.proj)+self.bias[None,:,:]
class TokenMask(tf.keras.layers.Layer):
    def __init__(self,rate): super().__init__(); self.rate=rate
    def call(self,x,training=None):
        if training: return tf.where(tf.random.uniform(tf.shape(x))<self.rate,tf.zeros_like(x),x)
        return x
class AddCLSPos(tf.keras.layers.Layer):
    def build(self,shape):
        self.cls=self.add_weight(shape=(1,1,int(shape[-1])),initializer='zeros'); self.pos=self.add_weight(shape=(1,int(shape[1])+1,int(shape[-1])),initializer=tf.keras.initializers.RandomNormal(stddev=.02))
    def call(self,x): return tf.concat([tf.repeat(self.cls,tf.shape(x)[0],axis=0),x],axis=1)+self.pos
def model(total,nraw,nder,d=96):
    ids=tf.keras.Input((nraw,),dtype='int32',name='ids'); cn=tf.keras.Input((nraw,),name='cn'); cm=tf.keras.Input((nraw,),name='cm'); dn=tf.keras.Input((nder,),name='dn'); dm=tf.keras.Input((nder,),name='dm')
    mid=TokenMask(.08)(ids); emb=tf.keras.layers.Embedding(total,d)(mid); rc=PLR(nraw,16,d)(cn); rd=PLR(nder,16,d)(dn)
    rc=tf.keras.layers.Multiply()([rc,tf.keras.layers.Reshape((nraw,1))(1-cm)]); rd=tf.keras.layers.Multiply()([rd,tf.keras.layers.Reshape((nder,1))(1-dm)])
    x=tf.keras.layers.Concatenate(axis=1)([emb+rc,rd]); x=AddCLSPos()(x)
    for i in range(3):
        a=tf.keras.layers.LayerNormalization()(x); a=tf.keras.layers.MultiHeadAttention(4,d//4,dropout=.10)(a,a); x=x+tf.keras.layers.Dropout(.10)(a)
        a=tf.keras.layers.LayerNormalization()(x); a=tf.keras.layers.Dense(2*d,activation='gelu')(a); a=tf.keras.layers.Dropout(.10)(a); a=tf.keras.layers.Dense(d)(a); x=x+a
    cls=tf.keras.layers.Lambda(lambda z:z[:,0,:])(x); out=tf.keras.layers.Dense(1,activation='sigmoid')(tf.keras.layers.Dense(d,activation='gelu')(tf.keras.layers.LayerNormalization()(cls)))
    m=tf.keras.Model([ids,cn,cm,dn,dm],out); m.compile(tf.keras.optimizers.AdamW(8e-4,weight_decay=2e-5),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='auc')]); return m
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED); tp,sp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); feats=[c for c in te if c!=ID]; y=tr[TARGET].to_numpy('float32'); rawtr=tr[feats]; rawte=te[feats]; dertr=derived(rawtr); derte=derived(rawte)
outer=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); tests=[]; fid=np.zeros(len(tr),'int8'); rows=[]
for fold,(a,b) in enumerate(outer.split(tr,y),1):
    fid[b]=fold; (li,lv,lt),total=lookup_arrays(rawtr.iloc[a],rawtr.iloc[b],rawte); (ci,cv,ct),(mi,mv,mt)=quantile_arrays(rawtr.iloc[a],rawtr.iloc[b],rawte); (di,dv,dt),(dmi,dmv,dmt)=quantile_arrays(dertr.iloc[a],dertr.iloc[b],derte)
    net=model(total,len(feats),dertr.shape[1]); ck=f'/kaggle/working/lookup_{fold}.weights.h5'; cb=[tf.keras.callbacks.EarlyStopping(monitor='val_auc',mode='max',patience=3,min_delta=2e-5,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint(ck,monitor='val_auc',mode='max',save_best_only=True,save_weights_only=True)]
    net.fit([li,ci,mi,di,dmi],y[a],validation_data=([lv,cv,mv,dv,dmv],y[b]),epochs=18,batch_size=2048,callbacks=cb,verbose=2)
    vp=net.predict([lv,cv,mv,dv,dmv],batch_size=8192,verbose=0).ravel(); oof[b]=vp; tests.append(net.predict([lt,ct,mt,dt,dmt],batch_size=8192,verbose=0).ravel()); rows.append({'fold':fold,'auc':roc_auc_score(y[b],vp)}); print(rows[-1]); tf.keras.backend.clear_session(); gc.collect()
pd.DataFrame({ID:tr[ID],'fold':fid,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_lookup_transformer.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:np.mean(tests,axis=0)}).to_csv('/kaggle/working/test_lookup_transformer.csv',index=False)
s=[r['auc'] for r in rows]; metrics={'concept_credit':'Tamerlan Omralinov lookup-transformer concept','implementation':'original fold-safe TensorFlow adaptation','official_data_only':True,'seed':SEED,'folds':FOLDS,'fold_auc':s,'fold_mean':float(np.mean(s)),'fold_std':float(np.std(s)),'oof_auc':roc_auc_score(y,oof),'submission_created':False,'device':[x.name for x in tf.config.list_physical_devices('GPU')]}; Path('/kaggle/working/metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
