# Attribution

Every file in this directory is a derivative of a publicly shared Kaggriculture
notebook, released by its author under Apache-2.0. The upstream licence text and
the authors' own attribution chains are retained verbatim inside each file; they
run to roughly 230 lines at the top of the larger ones.

The two base agents used here:

| our file | upstream notebook | author |
|---|---|---|
| `v11`, `v14`, `v16` | `aurax7/kaggriculture-shop-router-reactive-v5` (EXP257) | aurax7, building on a lineage crediting thomastschinkel, yhay81, destbreso, tetsutani, prvsiyan, Dmitrii Gluzdov and Ahmed Berat Ozer |
| `v17`, `v19`, `v21` | `haideptry/the-2965-master-hybrid-engine` | haideptry, same lineage |
| `v23`, `v25` | `ahmedberatozer/kaggriculture-v54-productive-wheat-and-patient` | Ahmed Berat Ozer, same lineage |

What is ours in these files is small and worth stating precisely. Against its
base, `v17` differs by 30 lines, 8 of which are comments: one setting flipped
from `False` to `True`, and a 17-line class. The others differ by a similar
amount. `git diff` against the upstream notebook is the honest way to see it.

The harness in the parent directory is ours.
