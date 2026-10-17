"""Fetch official MSS daily station records (weather.gov.sg) used ONLY to validate the data.gov.sg 5-minute dataset.

  https://www.weather.gov.sg/files/dailydata/DAILYDATA_<station>_<YYYYMM>.csv

Polite: sequential, one request at a time with a delay, browser User-Agent, cached under data/mss/raw/ (a cached file
or a cached "missing" marker is never fetched again). Every request is logged to data/mss/fetch.log.

  python src/mss_fetch.py probe            which of our station IDs have an MSS file (probe month 2023-01 and 2019-01)
  python src/mss_fetch.py fetch [S24 ...]  all months 2017-2024 for the given stations (default: probed ones)
"""
import datetime as _dt
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "mss")
RAW = os.path.join(D, "raw")
LOG = os.path.join(D, "fetch.log")
URL = "https://www.weather.gov.sg/files/dailydata/DAILYDATA_{s}_{ym}.csv"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) "
      "Version/17.0 Safari/605.1.15")
DELAY = 0.35
MONTHS = [f"{y}{m:02d}" for y in range(2017, 2025) for m in range(1, 13)]


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def get(s, ym):
    os.makedirs(RAW, exist_ok=True)
    p = os.path.join(RAW, f"DAILYDATA_{s}_{ym}.csv")
    miss = p + ".missing"
    if os.path.exists(p):
        return "cached"
    if os.path.exists(miss):
        return "cached-missing"
    req = urllib.request.Request(URL.format(s=s, ym=ym), headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            if b"Station" not in body[:200]:
                open(miss, "w").write("not a data file\n")
                log(f"{s} {ym} not-a-data-file {len(body)}")
                return "missing"
            open(p, "wb").write(body)
            log(f"{s} {ym} ok {len(body)}")
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(miss, "w").write("404\n")
                log(f"{s} {ym} 404")
                return "missing"
            log(f"{s} {ym} http {e.code} attempt {attempt}")
        except Exception as e:  # network hiccup: back off and retry
            log(f"{s} {ym} error {e!r} attempt {attempt}")
        time.sleep(3 * (attempt + 1))
    return "failed"


def our_stations():
    import csv
    return [r["station_id"] for r in csv.DictReader(open(os.path.join(ROOT, "outputs", "ledger", "stations.csv")))]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        found = []
        for s in our_stations():
            res = [get(s, ym) for ym in ("202301", "201901")]
            time.sleep(DELAY)
            ok = any(r in ("ok", "cached") for r in res)
            print(f"MSS probe {s}: {res}")
            if ok:
                found.append(s)
        json.dump(found, open(os.path.join(D, "stations_found.json"), "w"))
        print(f"MSS probe: {len(found)} of {len(our_stations())} of our station IDs have MSS daily files")
    elif cmd == "fetch":
        sts = sys.argv[2:] or json.load(open(os.path.join(D, "stations_found.json")))
        n = {"ok": 0, "missing": 0, "failed": 0, "cached": 0, "cached-missing": 0}
        for s in sts:
            for ym in MONTHS:
                r = get(s, ym)
                n[r] += 1
                if r in ("ok", "missing", "failed"):
                    time.sleep(DELAY)
            print(f"MSS fetch {s} done {n}", flush=True)
        print(f"MSS fetch total {n}")


if __name__ == "__main__":
    main()
