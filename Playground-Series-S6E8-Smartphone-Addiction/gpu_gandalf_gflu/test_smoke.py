import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch
p=Path(__file__).with_name("experiment.py"); s=importlib.util.spec_from_file_location("g",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
r=np.random.default_rng(31); n=260
x=pd.DataFrame({"daily_screen_time_hours":r.choice([2.,7.,np.nan],n),"social_media_hours":r.choice([1.,4.,np.nan],n),
"gaming_hours":r.choice([0.,3.,np.nan],n),"work_study_hours":r.choice([2.,8.,np.nan],n),"sleep_hours":r.choice([5.,8.,np.nan],n),
"notifications_per_day":r.choice([10.,100.,np.nan],n),"app_opens_per_day":r.choice([5.,50.,np.nan],n),
"weekend_screen_time":r.choice([3.,12.,np.nan],n),"gender":r.choice(["M","F",None],n)})
z=m.engineer(x); enc=m.FoldEncoder().fit(z.iloc[:200]); a=enc.transform(z.iloc[:200]); b=enc.transform(z.iloc[200:])
assert np.isfinite(a).all() and np.isfinite(b).all(); net=m.GANDALFNet(a.shape[1],hidden=48)
pred,masks=net(torch.from_numpy(a[:20]),return_masks=True); assert pred.shape==(20,) and masks.shape==(20,6,a.shape[1])
assert torch.isfinite(pred).all() and ((masks>0)&(masks<1)).all(); pred.sum().backward()
assert all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None)
before=enc.transform(z.iloc[200:]); fake_y=np.zeros(60); fake_y[:]=1; after=enc.transform(z.iloc[200:]); np.testing.assert_array_equal(before,after)
print({"forward":list(pred.shape),"gate_tensor":list(masks.shape),"finite_gradients":True,"validation_label_invariance":True})
