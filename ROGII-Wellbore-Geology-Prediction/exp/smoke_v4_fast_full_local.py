"""Execute every inference-only notebook cell locally with package path rewrites."""
from pathlib import Path
import json,os,time
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/"kernels/harshini_pilkwang_v4_fast/rogii-harshini-pilkwang-v4-fast.ipynb"
n=json.loads(p.read_text()); ns={"__name__":"v4_fast_smoke"}
out=ROOT/"exp/results/v4_fast_local"; out.mkdir(parents=True,exist_ok=True)
t0=time.time()
for i,c in enumerate(n["cells"]):
    if c["cell_type"]!="code": continue
    s="".join(c.get("source",[]))
    s=s.replace("@njit(cache=True)","@njit(cache=False)").replace(
        "@njit(cache=True, nogil=True)","@njit(cache=False, nogil=True)")
    if i==0:
        s=s.replace(
          'CANDS = ["/kaggle/input/rogii-wellbore-geology-prediction",\n'
          '         "/kaggle/input/competitions/rogii-wellbore-geology-prediction", "."]',
          f'CANDS = [r"{ROOT/"data"}"]')
    s=s.replace(
      ' _Path("/kaggle/input/datasets/boltuzamaki/rogii-harshini-fast-models"),',
      f' _Path(r"{ROOT/"exp/results/harshini_fast_model_package"}"),')
    s=s.replace(
      ' _Path("/kaggle/input/datasets/pilkwang/rogii-model-package"),',
      f' _Path(r"{ROOT/"exp/public_artifacts/pilkwang"}"),')
    s=s.replace(
      ' _Path("/kaggle/input/datasets/boltuzamaki/rogii-stack-v4-lgb7-package"),',
      f' _Path(r"{ROOT/"exp/results/stack_v4_lgb7_package"}"),')
    s=s.replace('"/kaggle/working/submission.csv"',
                f'r"{out/"submission.csv"}"')
    s=s.replace('"/kaggle/working/inference_audit.json"',
                f'r"{out/"inference_audit.json"}"')
    exec(compile(s,f"{p}:cell{i}","exec"),ns)
print("runtime_seconds",time.time()-t0)
