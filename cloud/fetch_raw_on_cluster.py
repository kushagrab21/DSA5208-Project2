"""Optional Dataproc stage: download the eight annual CSVs from data.gov.sg straight into the bucket, on the cluster's
driver, so the 7.5 GB never have to be uploaded from the laptop.

Uses the data.gov.sg open-data API exactly as .application/download_raw.sh did: initiate-download, poll-download until
a signed URL is returned, then stream the file into `hadoop fs -put - <gs path>`. One file at a time (polite).
Needs outbound internet from the master node (the cluster is created with --public-ip-address).

  gcloud dataproc jobs submit pyspark gs://.../cloud/fetch_raw_on_cluster.py ... -- --dest gs://BUCKET/p2/data/raw
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

API = "https://api-open.data.gov.sg/v1/public/api/datasets"
DATASETS = {2017: "d_1990a5a1aeaf3dd243cf4dae294a61c4", 2018: "d_024fb501ce7092b71bb713eaf54fa7eb",
            2019: "d_61995f092320e7155b7528050880b502", 2020: "d_9e7de44094f876f6804b8b5bcee45c81",
            2021: "d_3b41598f74f1f11fc3430348fea51af5", 2022: "d_42d64cc6c176ace1c52fbb40b9ede302",
            2023: "d_f864cc30d58b467db83659ad17c737bf", 2024: "d_a0b69d3e02576a1fd0ab673e71f83507"}


def get_json(url):
    # data.gov.sg answers 403 to urllib's default "Python-urllib/x.y" User-Agent (seen on the first Dataproc run).
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (DSA5208 coursework)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def signed_url(ds):
    get_json(f"{API}/{ds}/initiate-download")
    for _ in range(60):
        d = get_json(f"{API}/{ds}/poll-download").get("data") or {}
        if d.get("status") == "DOWNLOAD_SUCCESS" and d.get("url"):
            return d["url"]
        time.sleep(5)
    raise RuntimeError(f"no download URL for {ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="gs://BUCKET/p2/data/raw")
    ap.add_argument("--years", default="2017-2024")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    failed = []
    for y in range(lo, hi + 1):
        dst = f"{a.dest.rstrip('/')}/rainfall_{y}.csv"
        if subprocess.call(["hadoop", "fs", "-test", "-e", dst]) == 0:
            print(f"RAW {y} already in bucket: {dst}", flush=True)
            continue
        try:
            url = signed_url(DATASETS[y])
            curl = subprocess.Popen(["curl", "-s", "--fail", "--retry", "5", "--retry-delay", "5", url],
                                    stdout=subprocess.PIPE)
            put = subprocess.call(["hadoop", "fs", "-put", "-f", "-", dst], stdin=curl.stdout)
            curl.stdout.close()
            rc = curl.wait()
            if rc != 0 or put != 0:
                raise RuntimeError(f"curl rc={rc} put rc={put}")
            size = subprocess.check_output(["hadoop", "fs", "-du", "-s", dst], text=True).split()[0]
            print(f"RAW {y} done {size} bytes -> {dst}", flush=True)
        except Exception as e:
            failed.append(y)
            print(f"RAW {y} FAILED: {e!r}", flush=True)
        time.sleep(2)
    if failed:
        print(f"RAW FAILED years: {failed}")
        sys.exit(1)
    print("RAW ALL_DONE")


if __name__ == "__main__":
    main()
