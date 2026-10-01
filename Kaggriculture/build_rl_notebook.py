"""Build a private Kaggle GPU notebook containing the RL training package."""
import argparse,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
OUT=ROOT/("kaggle_rl_smoke" if args.smoke else "kaggle_rl"); OUT.mkdir(exist_ok=True)

def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s}
def md(s): return {"cell_type":"markdown","metadata":{},"source":s}

cells=[md("# Kaggriculture recurrent population self-play\n\nPrivate GPU training notebook. Checkpoints are saved as notebook output."),
       code("import os,sys,subprocess\nos.makedirs('rl',exist_ok=True)\nsubprocess.run([sys.executable,'-m','pip','install','-q','kaggle-environments==1.32.6'],check=True)\nprint('cpus',os.cpu_count())")]
for path in [ROOT/"main.py",*sorted((ROOT/"rl").glob("*.py"))]:
    rel=path.relative_to(ROOT).as_posix(); cells.append(code(f"%%writefile {rel}\n"+path.read_text()))
train_args=("--smoke --device cpu" if args.smoke else "--device cpu --actors 3 --updates 2000")
eval_args=("--seeds 1 --steps 48" if args.smoke else "--seeds 8")
train_cmd=f"subprocess.run([sys.executable,'-m','rl.train',*'{train_args}'.split()],check=True)"
eval_cmd=f"subprocess.run([sys.executable,'-m','rl.evaluate','rl_runs/checkpoints/latest.pt','--opponents','main.py',*'{eval_args}'.split(),'--out','rl_eval.json'],check=True)"
cells += [code(train_cmd),
          code(eval_cmd),
          code("import torch,glob,json\np=torch.load('rl_runs/checkpoints/latest.pt',map_location='cpu',weights_only=False)\nprint(p['update'],p.get('metrics'))\nprint(glob.glob('rl_runs/checkpoints/population/*.pt')[-5:])")]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
(OUT/"kaggriculture-rl-training.ipynb").write_text(json.dumps(nb,indent=1))
user=json.load(open(Path.home()/".kaggle/credentials.json"))["username"]
slug="kaggriculture-rl-smoke-test" if args.smoke else "kaggriculture-rl-training"
meta={"id":f"{user}/{slug}","title":"Kaggriculture RL Smoke Test" if args.smoke else "Kaggriculture RL Training","code_file":"kaggriculture-rl-training.ipynb","language":"python","kernel_type":"notebook","is_private":"true","enable_gpu":"false","enable_internet":"true","dataset_sources":[],"competition_sources":["kaggriculture"],"kernel_sources":[]}
(OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)); print(OUT)
