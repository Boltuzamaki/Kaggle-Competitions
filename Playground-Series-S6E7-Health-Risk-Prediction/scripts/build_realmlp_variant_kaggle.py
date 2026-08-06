"""Build a diversified RealMLP hyperparameter variant for Kaggle GPU."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"kaggle_kernels"/"realmlp_seed2027_gpu"/"realmlp_seed2027_gpu.ipynb"
OUT=ROOT/"kaggle_kernels"/"realmlp_variant7777_gpu"
def main():
 n=json.loads(SOURCE.read_text())
 replacements={
  "seed_everything(2027)":"seed_everything(7777)",
  "torch.cuda.manual_seed_all(2027)":"torch.cuda.manual_seed_all(7777)",
  '"dropout":       0.06':'"dropout":       0.04',
  '"pbld_freq_scale": 5.0':'"pbld_freq_scale": 3.5',
  '"pbld_lr_factor":  0.093':'"pbld_lr_factor":  0.075',
  '"lr":               0.01':'"lr":               0.008',
  '"weight_decay":     0.013':'"weight_decay":     0.010',
  '"ls_eps":       0.04':'"ls_eps":       0.03',
  '"epochs":    3':'"epochs":    4',
  '"random_state": 2027':'"random_state": 7777',
  "SEED = 2027":"SEED = 7777",
  'EXPERIMENT_ID = "realmlp_seed2027_gpu_7fold_3epoch"':'EXPERIMENT_ID = "realmlp_variant7777_gpu_7fold_4epoch"',
 }
 for a,b in replacements.items():
  count=0
  for cell in n["cells"]:
   source=cell.get("source",[])
   if isinstance(source,list):
    text="".join(source)
    hits=text.count(a)
    if hits:
     cell["source"]=text.replace(a,b).splitlines(keepends=True)
     count+=hits
   elif isinstance(source,str):
    hits=source.count(a)
    if hits:
     cell["source"]=source.replace(a,b)
     count+=hits
  if count!=1:raise RuntimeError((a,count))
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/"realmlp_variant7777_gpu.ipynb").write_text(json.dumps(n,indent=1))
 meta={"id":"boltuzamaki/health-risk-realmlp-variant-7777-gpu","title":"Health Risk RealMLP Variant 7777 GPU","code_file":"realmlp_variant7777_gpu.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":False,"dataset_sources":[],"kernel_sources":[],"competition_sources":["playground-series-s6e7"],"docker_image":"gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461","machine_shape":"NvidiaTeslaT4"}
 (OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__":main()
