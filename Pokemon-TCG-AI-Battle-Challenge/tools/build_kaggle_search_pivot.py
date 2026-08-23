"""Build three independent search-architecture live-field screens."""
from __future__ import annotations
import json
from pathlib import Path
from build_kaggle_lanes import SETUP, metadata, notebook

ROOT=Path(__file__).resolve().parents[1]; K=ROOT/'kaggle_remote'/'kernels'
JOBS=(
 ('cpu_search_beam4','ptcg-cpu-search-beam4-live-field','Beam width 4','beam4'),
 ('cpu_search_beam8','ptcg-cpu-search-beam8-live-field','Beam width 8','beam8'),
 ('cpu_search_ismcts','ptcg-cpu-search-ismcts-live-field','Imperfect information MCTS','ismcts'),
 ('cpu_search_gated_ismcts','ptcg-cpu-search-gated-ismcts-live-field','Gated imperfect information MCTS','gatedismcts'),
 ('cpu_search_beam_response','ptcg-cpu-search-beam-response-live-field','Opponent response beam','beam2response'),
 ('cpu_search_adaptive','ptcg-cpu-search-adaptive-live-field','Adaptive budget search','adaptivebudget'),
 ('cpu_search_beam2','ptcg-cpu-search-beam2-live-field','Beam width 2','beam2'),
 ('cpu_search_flatmc','ptcg-cpu-search-flatmc-live-field','Flat Monte Carlo','flatmc'),
 ('cpu_search_gatedflat','ptcg-cpu-search-gatedflat-live-field','Gated flat Monte Carlo','gatedflat'),
)
RUN='''from pathlib import Path
import json, subprocess, sys
out=Path("/kaggle/working/{candidate}")
cmd=[sys.executable,"automation/jobs/live_deck_gauntlet.py","--candidate","{candidate}","--games-per-deck","12","--max-decks","20","--deck-catalog","data/live_decks.json","--out",str(out)]
r=subprocess.run(cmd,check=False)
if r.returncode: raise RuntimeError("screen failed "+str(r.returncode))
report=json.loads(out.with_suffix(".json").read_text())
summary={{"experiment":"SEARCH-PIVOT-{candidate}","candidate":report["candidate"],"games":report["total_games"],"wins":report["wins"],"losses":report["losses"],"winrate":report["winrate"],"failures":report["candidate_failures"]+report["game_errors"],"submission_performed":False}}
Path("/kaggle/working/final_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))'''

for folder,kernel,title,candidate in JOBS:
    target=K/folder; target.mkdir(parents=True,exist_ok=True)
    target.joinpath('kernel-metadata.json').write_text(json.dumps(metadata(kernel,'PTCG CPU '+title,False),indent=2))
    target.joinpath('notebook.ipynb').write_text(json.dumps(notebook([('markdown','# '+title),('code',SETUP),('code',RUN.format(candidate=candidate))]),indent=1))
    print(target)
