# IEEE paper — Overleaf project

Everything needed to build the paper. Upload this whole folder to Overleaf
(New Project → Upload Project → zip this directory), set the compiler to
**pdfLaTeX**, and build `main.tex`.

```
main.tex                 the paper
refs.bib                 bibliography
figures/*.pdf            six figures, copied from ../../figures/
check_tex.py             structural validation (labels, cites, envs, tables)
verify_tex_numbers.py    checks every table against results/*.json
```

## Before you submit

1. **Check the citations.** `refs.bib` groups entries by confidence.
   Group A is verified. Group B are arXiv preprints and are cited as
   preprints on purpose — MuJoCo Playground and Humanoid-Gym are *not*
   conference-published, and citing them as if they were is an easy
   mistake. Group C needs volume/page numbers confirmed.

2. **Check the length.** The full version is ~8–9 pages. Most IEEE
   conferences allow 6 plus one for references. `main.tex` opens with a
   trimming guide listing what to cut, in order, and what not to touch.

3. **Fix the author block.** Affiliation and email are filled in; add
   co-authors or an advisor if needed.

## Re-checking after edits

```bash
python3 check_tex.py            # 0 errors expected
python3 verify_tex_numbers.py   # 58 checks, all passing
```

`check_tex.py` catches undefined references, missing citation keys,
missing figure files, unbalanced environments, and tables whose rows do
not match their column spec. It cannot catch layout problems — overfull
lines, a table wider than a column — so read the first real build.

`verify_tex_numbers.py` re-reads every number in the tables from the
result JSONs, including the derived ones (the 23.9× agreement ratio,
Warp's per-step convergence ratios, the 26× gap at the finest timestep).
Run it after any edit that touches a number.

## Regenerating the figures

The PDFs here are copies. To rebuild from data, run the plotting scripts
in the repository root and copy them across:

```bash
cd ../..
python3 tools/plot_convergence.py       # Figure 1
python3 tools/plot_crossembodiment.py   # Figure 2
python3 tools/plot_policy_null.py       # Figure 3
python3 tools/plot_tolerance.py         # Figure 4
python3 tools/plot_groundtruth.py       # Figure 5
python3 tools/plot_schematic.py         # Figure 6
cp figures/fig*.pdf paper/ieee/figures/
```
