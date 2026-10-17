"""Build the Canvas submission: Project_2/submission/DSA5208_Project2_Code.zip and DSA5208_Project2_Report.pdf.

The zip holds the code, configuration, checks, report source with its generated numbers and the figures, the small
output tables, the env.json of every run, the two trained models, the test fixtures (real files only), and the
Dataproc run (cloud_run/) once it is complete. It never holds data/, .venv/, caches, .application/ (the build's private
working notes), symlinks, or any file that looks like a credential. A SHA256SUMS.txt over every file is added at the root.
Usage: python tools/make_submission.py
"""
import hashlib
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), "submission")
ZIP = os.path.join(OUT, "DSA5208_Project2_Code.zip")
PDF = os.path.join(ROOT, "report", "DSA5208_Project2_Report.pdf")
TOP = "DSA5208_Project2"                       # folder name inside the zip
FILES = ["README.md", "requirements.txt", "env.sh"]
DIRS = ["src", "checks", "config", "cloud", "tools", "report", "figures", "outputs", "runs", "models", "tests"]
SKIP_DIRS = {"__pycache__", ".venv", ".application", ".spark_tmp", ".ipynb_checkpoints"}   # top-level data/ is never walked
SKIP_PATHS = (os.path.join("tests", "fixtures", "cloud_parity"),)   # copies of outputs/ that checks/qualify.py recreates
SKIP_FILES = re.compile(r"(\.pyc|\.DS_Store)$")
MAX_TABLE_BYTES = 5 * 2 ** 20                  # outputs/: small tables only
SECRET = re.compile(rb"(-----BEGIN [A-Z ]*PRIVATE KEY-----|ya29\.[0-9A-Za-z_\-]{20,}|AIza[0-9A-Za-z_\-]{30,}|"
                    rb"\"private_key\"|aws_secret_access_key|xox[bp]-[0-9A-Za-z\-]{10,})")


def cloud_complete():
    d = os.path.join(ROOT, "cloud_run")
    return os.path.exists(os.path.join(d, "cluster.json")) and os.path.isdir(os.path.join(d, "runs"))


def walk(rel_dir):
    base = os.path.join(ROOT, rel_dir)
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not os.path.islink(os.path.join(dp, d))
                  and not os.path.relpath(os.path.join(dp, d), ROOT).startswith(SKIP_PATHS)]
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if os.path.islink(p) or SKIP_FILES.search(fn):
                continue
            yield os.path.relpath(p, ROOT)


def main():
    dirs = list(DIRS) + (["cloud_run"] if cloud_complete() else [])
    files, skipped = list(FILES), []
    for d in dirs:
        for rel in walk(d):
            size = os.path.getsize(os.path.join(ROOT, rel))
            if rel.startswith("outputs" + os.sep) and size > MAX_TABLE_BYTES:
                skipped.append((rel, size))
                continue
            files.append(rel)
    bad = []
    for rel in files:
        if rel == os.path.join("tools", "make_submission.py"):
            continue            # this file contains the patterns themselves
        if SECRET.search(open(os.path.join(ROOT, rel), "rb").read()):
            bad.append(rel)
    if bad:
        print("SUBMISSION REFUSED: possible credentials in", bad)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    sums = []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in sorted(files):
            p = os.path.join(ROOT, rel)
            z.write(p, f"{TOP}/{rel}")
            sums.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {rel}")
        z.writestr(f"{TOP}/SHA256SUMS.txt", "\n".join(sums) + "\n")
    shutil.copy2(PDF, os.path.join(OUT, "DSA5208_Project2_Report.pdf"))
    zs = os.path.getsize(ZIP)
    by_top = {}
    for rel in files:
        k = rel.split(os.sep)[0]
        by_top[k] = by_top.get(k, 0) + os.path.getsize(os.path.join(ROOT, rel))
    print(f"SUBMISSION zip={ZIP} size={zs / 2 ** 20:.1f} MiB files={len(files) + 1} (incl. SHA256SUMS.txt)")
    print("SUBMISSION uncompressed by folder: " + ", ".join(f"{k} {v / 2 ** 20:.1f} MiB" for k, v in
                                                           sorted(by_top.items(), key=lambda kv: -kv[1])))
    print(f"SUBMISSION cloud_run included: {'yes' if 'cloud_run' in dirs else 'no (not complete yet)'}")
    for rel, size in skipped:
        print(f"SUBMISSION skipped large output {rel} ({size / 2 ** 20:.1f} MiB)")
    print(f"SUBMISSION pdf={os.path.join(OUT, 'DSA5208_Project2_Report.pdf')} "
          f"size={os.path.getsize(PDF) / 2 ** 20:.1f} MiB")


if __name__ == "__main__":
    main()
