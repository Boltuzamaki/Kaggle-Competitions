import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

p=Path(__file__).with_name('experiment.py'); s=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
rng=np.random.default_rng(9); n=250
cols=['daily_screen_time_hours','sleep_hours','notifications_per_day','app_opens_per_day','social_media_hours','gaming_hours','weekend_screen_time']
x=pd.DataFrame({c:rng.choice([1.,2.,3.,np.nan],n) for c in cols}); x['stress_level']=rng.choice(['Low','High',None],n); x['academic_work_impact']=rng.choice(['Low','High',None],n)
xt=x.iloc[:23].reset_index(drop=True); y=np.tile([0,1],n//2).astype('float32'); k,kt=m.make_keys(x),m.make_keys(xt)
fit,val=next(StratifiedKFold(5,shuffle=True,random_state=m.SEED).split(x,y)); a=m.fold_evidence(k,kt,y,fit,val,1); yc=y.copy(); yc[val]=1-yc[val]; b=m.fold_evidence(k,kt,yc,fit,val,1)
for left,right in zip(a,b): np.testing.assert_array_equal(left,right)
yf=y[fit]; ia,ib=next(StratifiedKFold(m.INNER,shuffle=True,random_state=m.SEED+100).split(fit,yf)); pos=int(ib[0]); yy=yf.copy(); yy[pos]=1-yy[pos]; c=k.columns[0]
u=m.map_evidence(k.iloc[fit[ia]][c],k.iloc[[fit[pos]]][c],yf[ia],24.); v=m.map_evidence(k.iloc[fit[ia]][c],k.iloc[[fit[pos]]][c],yy[ia],24.)
for left,right in zip(u,v): np.testing.assert_array_equal(left,right)
net=m.Net([7,8,5],4,d=32).eval()
with torch.no_grad(): z=net(torch.stack([torch.randint(0,7,(6,)),torch.randint(0,8,(6,)),torch.randint(0,5,(6,))],1),torch.randn(6,3),torch.zeros(6,3),torch.randn(6,4,2))
assert z.shape==(6,) and torch.isfinite(z).all()
print({'outer_validation_label_invariance':True,'own_label_exclusion':True,'inner_oof_coverage':True,'gated_forward':list(z.shape),'evidence_keys':k.shape[1]})
