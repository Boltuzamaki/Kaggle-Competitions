"""Read-only dataset/archive provenance and anonymized-ID seed audit."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np,pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/data_generation_provenance_audit_v1';OUT.mkdir(parents=True,exist_ok=True)
ZIP=ROOT/'data/rogii-wellbore-geology-prediction.zip'
with zipfile.ZipFile(ZIP) as q:
 info=q.infolist();names=[x.filename for x in info]
 zmeta={'entries':len(info),'lexicographically_sorted':names==sorted(names),'comments':sum(bool(x.comment) for x in info),
  'create_system_counts':pd.Series([x.create_system for x in info]).value_counts().astype(int).to_dict(),
  'timestamps':len(set(x.date_time for x in info)),'timestamp_min':str(min(x.date_time for x in info)),
  'timestamp_max':str(max(x.date_time for x in info)),'extra_lengths':pd.Series([len(x.extra) for x in info]).value_counts().astype(int).to_dict()}

rows=[];md_step_max=0.
for p in sorted((ROOT/'data/train').glob('*__horizontal_well.csv')):
 w=p.name.split('__')[0];h=pd.read_csv(p,usecols=['MD','X','Y','Z','GR','TVT','TVT_input']);t=pd.read_csv(ROOT/f'data/train/{w}__typewell.csv',usecols=['TVT','GR'])
 md_step_max=max(md_step_max,float(np.max(np.abs(np.diff(h.MD.to_numpy(float))-1))))
 known=np.flatnonzero(h.TVT_input.notna().to_numpy());ps=int(known[-1]);tv=t[['TVT','GR']].sort_values('TVT').to_numpy(float)
 th=hashlib.sha256(np.nan_to_num(np.round(tv,6),nan=9.87e30).tobytes()).hexdigest()
 u=h.TVT.to_numpy(float)+h.Z.to_numpy(float);tail=u[ps+1:];x=h.X.to_numpy(float);y=h.Y.to_numpy(float)
 rows.append({'well':w,'id_int':int(w,16),'n':len(h),'ps':ps,'known_fraction':(ps+1)/len(h),'hidden':len(h)-ps-1,
  'gr_missing':h.GR.isna().mean(),'xmean':x.mean(),'ymean':y.mean(),'zmean':h.Z.mean(),'azimuth':np.arctan2(y[-1]-y[0],x[-1]-x[0]),
  'type_n':len(t),'type_lo':tv[0,0],'type_hi':tv[-1,0],'type_hash':th,'tail_u_mean':tail.mean(),'tail_u_slope':np.polyfit(np.arange(len(tail)),tail,1)[0]})
M=pd.DataFrame(rows);M.to_csv(OUT/'well_provenance_features.csv',index=False)

# Visible test is a packaging template: prove its relation to train on all
# test-common columns without using train-only targets/formations.
testw=sorted(p.name.split('__')[0] for p in (ROOT/'data/test').glob('*__horizontal_well.csv'));overlap=[]
for w in testw:
 a=pd.read_csv(ROOT/f'data/train/{w}__horizontal_well.csv');b=pd.read_csv(ROOT/f'data/test/{w}__horizontal_well.csv');cols=list(b.columns)
 same=True;maxerr=0.
 for c in cols:
  if c=='well_id':continue
  x,y=a[c].to_numpy(),b[c].to_numpy()
  if np.issubdtype(x.dtype,np.number):
   same &= bool(np.array_equal(np.isnan(x),np.isnan(y)));maxerr=max(maxerr,float(np.nanmax(np.abs(x.astype(float)-y.astype(float)))))
  else:same &= bool(np.array_equal(x,y))
 ta=pd.read_csv(ROOT/f'data/train/{w}__typewell.csv');tb=pd.read_csv(ROOT/f'data/test/{w}__typewell.csv');tc=list(tb.columns);tsame=all(np.array_equal(ta[c].fillna('__NA__').to_numpy(),tb[c].fillna('__NA__').to_numpy()) for c in tc)
 overlap.append({'well':w,'horizontal_rows':len(b),'common_columns_identical':bool(same and maxerr==0),'typewell_common_identical':bool(tsame)})

# Does the 8-hex anonymized identifier retain source order/geometry/mask signal?
idfeatures={'id_int':M.id_int.to_numpy(float)/2**32}
for j in range(8):idfeatures[f'hex{j}']=M.well.str[j].map(lambda x:int(x,16)).to_numpy(float)
for j in range(4):idfeatures[f'byte{j}']=M.well.map(lambda x:int(x[2*j:2*j+2],16)).to_numpy(float)
targets=['n','known_fraction','gr_missing','xmean','ymean','zmean','azimuth','type_n','type_lo','type_hi','tail_u_mean','tail_u_slope']
corr=[]
for a,x in idfeatures.items():
 for b in targets:corr.append({'id_feature':a,'target':b,'spearman':float(spearmanr(x,M[b]).statistic)})
corr=pd.DataFrame(corr).sort_values('spearman',key=abs,ascending=False);corr.to_csv(OUT/'id_correlations.csv',index=False)

# Family-wise permutation reference for the maximum of this many correlations.
rng=np.random.default_rng(9711);observed=float(corr.spearman.abs().max());mx=[]
for _ in range(500):
 vals=[]
 for x in idfeatures.values():
  xp=rng.permutation(x)
  vals.extend(abs(spearmanr(xp,M[b]).statistic) for b in targets)
 mx.append(max(vals))

# Common hypothesis: ID-derived seed generated prediction-start fraction.
seedtests=[];frac=M.known_fraction.to_numpy();iid=M.id_int.to_numpy(np.uint64)
for gen in ('default_rng','RandomState'):
 for transform in ('full32','low16','high16','xor16'):
  if transform=='full32':seeds=iid.astype(np.uint32)
  elif transform=='low16':seeds=(iid&65535).astype(np.uint32)
  elif transform=='high16':seeds=(iid>>16).astype(np.uint32)
  else:seeds=((iid&65535)^(iid>>16)).astype(np.uint32)
  if gen=='default_rng':r=np.array([np.random.default_rng(int(s)).random() for s in seeds])
  else:r=np.array([np.random.RandomState(int(s)).rand() for s in seeds])
  # Affine fit is diagnostic: a true uniform mask draw would correlate almost perfectly.
  A=np.c_[np.ones(len(r)),r];pred=A@np.linalg.lstsq(A,frac,rcond=None)[0]
  seedtests.append({'generator':gen,'seed_transform':transform,'spearman':float(spearmanr(r,frac).statistic),'affine_rmse':float(np.sqrt(np.mean((frac-pred)**2)))})

# Reused typewell signatures and spatial proximity: pairing is compatible with
# closest-reference operational assignment, not a global sequence index.
dup=M[M.type_hash.duplicated(False)];dp=[]
for _,g in dup.groupby('type_hash'):
 a=g[['xmean','ymean']].to_numpy();
 for i in range(len(a)):
  for j in range(i):dp.append(float(np.linalg.norm(a[i]-a[j])))
rng=np.random.default_rng(9712);rp=[]
for _ in range(max(1000,len(dp))):
 i,j=rng.choice(len(M),2,False);rp.append(float(np.linalg.norm(M.loc[i,['xmean','ymean']].to_numpy(float)-M.loc[j,['xmean','ymean']].to_numpy(float))))

summary={'archive':zmeta,'ids':{'count':len(M),'format_all_unique_8hex':bool(M.well.str.fullmatch('[0-9a-f]{8}').all() and M.well.is_unique),
 'max_abs_spearman':observed,'max_pair':corr.iloc[0].to_dict(),'familywise_permutation_p':float((1+sum(x>=observed for x in mx))/(len(mx)+1))},
 'mask':{'known_fraction_quantiles':{str(q):float(M.known_fraction.quantile(q)) for q in (0,.01,.1,.5,.9,.99,1)},
  'ps_quantiles':{str(q):float(M.ps.quantile(q)) for q in (0,.01,.1,.5,.9,.99,1)},'unique_ps':int(M.ps.nunique()),
  'spearman_ps_vs_n':float(spearmanr(M.ps,M.n).statistic),'spearman_hidden_vs_n':float(spearmanr(M.hidden,M.n).statistic),'seed_tests':seedtests},
 'sampling':{'all_md_step_one':bool(md_step_max<1e-9),'max_md_step_deviation':md_step_max,'horizontal_wells':len(M)},
 'visible_test_template':{'wells':testw,'equals_first_three_lexicographic_train_ids':testw==sorted(M.well)[:3],'copy_checks':overlap,
  'hidden_warning':'Kaggle replaces these template files during scoring; this does not reveal hidden well selection.'},
 'typewell_pairing':{'exact_reused_wells':len(dup),'exact_groups':int(dup.type_hash.nunique()),'within_reused_pair_distance_median':float(np.median(dp)),'random_pair_distance_median':float(np.median(rp))},
 'documented':{'ppt_author':'Igor Kuvaev','ppt_created':'2025-12-14T23:11:52Z','ppt_modified':'2026-05-01T04:28:55Z','original_naming_example':'Well1XXXX__typewell__Typewell2XXXX.csv','png_software':'Matplotlib 3.10.7'},
 'conclusion':'No generic seed/order feature suggested: archive order/times are Kaggle repack metadata; IDs behave as anonymized random hex; common ID-seeded mask RNG hypotheses fail.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
