"""Enforced release check for the report (circuit D5). Refuses a release unless:

  - every requirement E1-E14 is mapped to a report section by a \\req{E#} marker placed in that section;
  - the body has no hand-typed numbers: every number comes from a generated macro or table (report/generated/);
  - the authors' names and student IDs are present on the title page;
  - an AI-use declaration section is present and names the tools.

A "hand-typed number" is any digit in the document body outside these contexts: comments; \\label/\\ref/\\eqref/
\\cite keys; \\input/\\include/\\includegraphics arguments and options; length arguments (\\vspace, widths like
0.48\\textwidth); tabular column specs; \\texttt{} (file and script names); the \\author block; the brief's task names "Task 1"/"Task 2"; student IDs. Generated
tables are \\input files, so their numbers are not in the body.
Identifiers are not measurements and may be typed: the data's calendar years 2017-2024, station IDs (S24),
lecture numbers, student IDs. Every \\val{key} used must be defined in report/generated/values.tex.
Usage: python checks/requirements_check.py [report/DSA5208_Project2_Report.tex]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEX = os.path.join(ROOT, "report", "DSA5208_Project2_Report.tex")
REQS = {
    "E1": "All 8 annual datasets 2017-2024 used", "E2": "Spark for processing and ML",
    "E3": "Cleaning explained with counts", "E4": "Patterns within a year", "E5": "Changes across 2017-2024, coverage-adjusted",
    "E6": "Visualisations", "E7": "P(rain next 30 min > theta) from (station, time, theta)",
    "E8": "Features selected and justified", "E9": "Evaluated with MLlib on unseen time",
    "E10": "Computing environment, captured from the run", "E11": "AI use declared", "E12": "Limitations / insights",
    "E13": "Code + README, trained model", "E14": "Names + student IDs"}
AUTHORS = [("Kushagra Bhatnagar", "A0351591A"), ("Pratyush Sinha", "A0352482A")]
AI_TOOLS = ["Claude Code"]

STRIP = [
    r"(?<!\\)%.*",                                                     # comments
    r"\\(?:label|ref|eqref|cref|Cref|autoref|cite|citep|citet|req|url|href)\{[^}]*\}",
    r"\\(?:input|include|includegraphics|graphicspath)(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\(?:vspace|hspace|setlength|addtolength|rule|resizebox|scalebox)\*?(?:\{[^}]*\})+",
    r"\\begin\{(?:tabular|tabularx|longtable|array)\}(?:\{[^}]*\})+",
    r"\\begin\{(?:figure|table)\}\[[^\]]*\]",
    r"\[[^\]]*(?:width|height|scale)[^\]]*\]",
    r"[0-9.]+\\(?:textwidth|linewidth|columnwidth|textheight)",
    r"\\(?:texttt|code|src)\{(?:[^{}]|\{[^{}]*\})*\}",                   # code, file and script names
    r"\bDSA5208\b", r"\bProject [12]\b", r"\bSHA-256\b", r"\bOpus 5\.5\b",     # course, algorithm, AI model names
    r"\\\\\[[0-9.]+(?:pt|ex|em|mm)\]",                                      # line-break spacing \\[3pt]
    r"\$y=1\$", r"1\+\\theta", r"_\{30\}",                                  # notation
    r"\\author\{(?:[^{}]|\{[^{}]*\})*\}",
    r"\\(?:small|footnotesize|scriptsize)\b",
    r"\\val\{[^}]*\}",                                                 # generated values: \val{key}
    r"\\(?:begin|end)\{[^}]*\}",
    r"\bTask [12]\b",                                                 # the brief's task names
    r"\bLecture [0-9]\b",                                             # course lecture numbers (citations)
    r"\b20(?:1[7-9]|2[0-4])\b",                                        # calendar years of the data (identifiers)
    r"\bS[0-9]{2,3}\b",                                               # station identifiers
    r"\b[A-Z]\d{7}[A-Z]\b",                                           # NUS student IDs (checked separately)
]


def body_of(tex):
    i = tex.find("\\begin{document}")
    j = tex.find("\\end{document}")
    return tex[i:j] if i >= 0 and j > i else tex


def hand_typed_numbers(tex):
    b = body_of(tex)
    for pat in STRIP:
        b = re.sub(pat, " ", b)
    # macro names never contain digits in LaTeX; remaining digits are literal text
    hits = []
    for ln, line in enumerate(b.splitlines(), 1):
        for m in re.finditer(r"\d+(?:[.,]\d+)*", line):
            hits.append((ln, m.group(0), line.strip()[:90]))
    return hits


def section_map(tex):
    out = {}
    cur = "(before first section)"
    for line in body_of(tex).splitlines():
        m = re.search(r"\\(?:sub)*section\*?\{([^}]*)\}", line)
        if m:
            cur = m.group(1)
        for r in re.findall(r"\\req\{(E\d+)\}", line):
            out.setdefault(r, []).append(cur)
    return out


def main():
    p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEX
    tex = open(p).read()
    problems = []
    sm = section_map(tex)
    for k, v in REQS.items():
        where = "; ".join(sorted(set(sm.get(k, [])))) or "UNMAPPED"
        print(f"REQ {k:<3} {v:<48} -> {where}")
        if k not in sm:
            problems.append(f"{k} unmapped")
    vals = os.path.join(os.path.dirname(os.path.abspath(p)), "generated", "values.tex")
    used = set(re.findall(r"\\val\{([^}]*)\}", body_of(tex)))
    defined = set(re.findall(r"\\setval\{([^}]*)\}", open(vals).read())) if os.path.exists(vals) else set()
    undefined = sorted(used - defined)
    print(f"REQ generated values used: {len(used)}; undefined: {len(undefined)}" +
          (f" {undefined[:10]}" if undefined else ""))
    if undefined:
        problems.append(f"{len(undefined)} \\val keys not defined in generated/values.tex")
    hits = hand_typed_numbers(tex)
    print(f"REQ hand-typed numbers in body: {len(hits)}")
    for ln, num, ctx in hits[:20]:
        print(f"    body line {ln}: {num!r} in: {ctx}")
    if hits:
        problems.append(f"{len(hits)} hand-typed numbers")
    for name, sid in AUTHORS:
        ok = name in tex and sid in tex
        print(f"REQ author {name} ({sid}): {'present' if ok else 'MISSING'}")
        if not ok:
            problems.append(f"author {name} {sid} missing")
    ai = re.search(r"\\section\*?\{[^}]*(AI|artificial intelligence)[^}]*\}", tex, re.I)
    tools = all(t in tex for t in AI_TOOLS)
    print(f"REQ AI declaration section: {'present' if ai else 'MISSING'}; tools named: {tools}")
    if not ai or not tools:
        problems.append("AI declaration missing or incomplete")
    for q in problems:
        print("REQ PROBLEM:", q)
    if problems:
        print("REQUIREMENTS CHECK: REFUSED (no release)")
        sys.exit(1)
    print("REQUIREMENTS CHECK: PASS")


if __name__ == "__main__":
    main()
