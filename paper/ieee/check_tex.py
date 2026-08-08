#!/usr/bin/env python3
"""Static checks on main.tex, standing in for a compiler we do not have.

Catches the failure modes that actually break a first Overleaf build:
undefined references, missing citation keys, missing figure files, and
unbalanced environments. It cannot catch layout problems (overfull boxes,
tables wider than a column), so the first real build still matters.
"""
import pathlib
import re
import sys

tex = pathlib.Path("main.tex").read_text()
bib = pathlib.Path("refs.bib").read_text()
problems = []

# Strip comments so commented-out examples are not treated as real usage.
body = re.sub(r"(?<!\\)%.*", "", tex)

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
refs = set(re.findall(r"\\ref\{([^}]*)\}", body))
for r in sorted(refs - labels):
    problems.append(f"\\ref{{{r}}} has no matching \\label")
for l in sorted(labels - refs):
    problems.append(f"NOTE: \\label{{{l}}} is never referenced")

bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for grp in re.findall(r"\\cite\{([^}]*)\}", body):
    cited.update(k.strip() for k in grp.split(","))
for c in sorted(cited - bibkeys):
    problems.append(f"\\cite{{{c}}} is not in refs.bib")
for b in sorted(bibkeys - cited):
    problems.append(f"NOTE: refs.bib entry '{b}' is never cited")

for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
    if not (pathlib.Path("figures") / g).exists() and not pathlib.Path(g).exists():
        problems.append(f"missing figure file: {g}")

envs = re.findall(r"\\(begin|end)\{([^}]*)\}", body)
stack = []
for kind, name in envs:
    if kind == "begin":
        stack.append(name)
    else:
        if not stack:
            problems.append(f"\\end{{{name}}} with nothing open")
        elif stack[-1] != name:
            problems.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
        else:
            stack.pop()
for name in stack:
    problems.append(f"\\begin{{{name}}} never closed")

if body.count("{") != body.count("}"):
    problems.append(f"brace imbalance: {body.count('{')} open, {body.count('}')} close")

def brace_group(s, i):
    """Return (contents, index_after) for the {...} starting at s[i], counting
    depth. A flat regex truncates specs like p{0.6\\columnwidth}cc at the
    inner brace and miscounts every column."""
    assert s[i] == "{"
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced")


tabulars = []
for m in re.finditer(r"\\begin\{tabular\}", body):
    spec, after = brace_group(body, m.end())
    end = body.find(r"\end{tabular}", after)
    tabulars.append((spec, body[after:end]))

for spec, content in tabulars:
    clean = re.sub(r"@\{[^}]*\}", "", spec)
    clean = re.sub(r"[pmb]\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "P", clean)
    ncol = len(re.findall(r"[lcrP]", clean))
    for line in content.split(r"\\"):
        line = line.strip()
        if not line or line.startswith("\\") or "&" not in line:
            continue
        if "multicolumn" in line:
            continue
        n = len(line.split("&"))
        if n != ncol:
            problems.append(f"tabular row has {n} cells, spec declares {ncol}: {line[:60]}...")

hard = [p for p in problems if not p.startswith("NOTE:")]
for p in problems:
    print(("  " if p.startswith("NOTE:") else "  ERROR ") + p)
print(f"\n{len(hard)} error(s), {len(problems)-len(hard)} note(s)")
sys.exit(1 if hard else 0)
