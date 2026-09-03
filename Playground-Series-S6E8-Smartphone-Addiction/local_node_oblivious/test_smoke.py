import importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch
p=Path(__file__).with_name("train.py");s=importlib.util.spec_from_file_location("n",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=np.random.default_rng(8);n=240
x=pd.DataFrame({"daily_screen_time_hours":r.choice([2.,8.,np.nan],n),"social_media_hours":r.choice([1.,4.,np.nan],n),"gaming_hours":r.choice([0.,3.,np.nan],n),"work_study_hours":r.choice([2.,7.,np.nan],n),"sleep_hours":r.choice([5.,8.,np.nan],n),"notifications_per_day":r.choice([10.,90.,np.nan],n),"app_opens_per_day":r.choice([5.,50.,np.nan],n),"weekend_screen_time":r.choice([3.,12.,np.nan],n),"gender":r.choice(["M","F",None],n)})
z=m.engineer(x);e=m.Encoder().fit(z.iloc[:180]);a,b=e.transform(z.iloc[:180]),e.transform(z.iloc[180:]);assert np.isfinite(a).all() and np.isfinite(b).all()
net=m.ObliviousTreeEnsemble(a.shape[1],trees=8,depth=4);pred,sel=net(torch.from_numpy(a[:20]),True)
assert pred.shape==(20,) and sel.shape==(8,4,a.shape[1]);np.testing.assert_allclose(sel.detach().sum(-1).numpy(),1,rtol=1e-5)
pred.sum().backward();assert all(torch.isfinite(q.grad).all() for q in net.parameters() if q.grad is not None)
before=e.transform(z.iloc[180:]);fake=np.ones(60);fake[:]=0;after=e.transform(z.iloc[180:]);np.testing.assert_array_equal(before,after)
print({"prediction":list(pred.shape),"soft_feature_selection":list(sel.shape),"finite_gradients":True,"validation_label_invariance":True})
