#!/usr/bin/env bash
# One Dataproc run of the whole pipeline up to Task 2 validation (the 2024 test is evaluated once, locally).
#
# THIS SCRIPT CREATES BILLED GOOGLE CLOUD RESOURCES. It is written by the builder and run only by the owner of the
# Google Cloud project (the parent session / the user), after `gcloud auth login` and choosing a project with billing.
# Steps are separate subcommands so each can be run, checked and repeated on its own:
#
#   cloud/dataproc_run.sh plan       print every command and the expected cost; creates nothing
#   cloud/dataproc_run.sh bucket     create the bucket and upload the code (and, unless P2_FETCH_ON_CLUSTER=1, the
#                                    8 raw CSVs, 7.5 GB, from this machine)
#   cloud/dataproc_run.sh cluster    create the cluster (shape below) and save its description
#   cloud/dataproc_run.sh rawfetch   OPTIONAL instead of uploading: download the CSVs from data.gov.sg on the cluster
#                                    straight into the bucket (cloud/fetch_raw_on_cluster.py)
#   cloud/dataproc_run.sh jobs       submit the stages in order; stops at the first failed stage
#   cloud/dataproc_run.sh fetch      copy outputs/, runs/ (env.json per stage) and models/ back to cloud_run/
#   cloud/dataproc_run.sh delete     DELETE the cluster (do this as soon as `jobs` ends; Lecture 6: ephemeral clusters)
#
# Settings (override with environment variables):
set -euo pipefail
PROJECT="${P2_GCP_PROJECT:?set P2_GCP_PROJECT to the Google Cloud project id}"
REGION="${P2_REGION:-asia-southeast1}"                 # Singapore
CLUSTER="${P2_CLUSTER:-dsa5208-p2}"
BUCKET="${P2_BUCKET:-gs://${PROJECT}-dsa5208-p2}"
IMAGE="${P2_IMAGE:-3.0-debian13}"                      # Lecture 6's image: Spark 4.1.2, Python 3.12, Java 21
                                                       # (local runs: Spark 4.0.4, Python 3.11, Java 17)
MASTER_TYPE="${P2_MASTER_TYPE:-n2-standard-4}"
WORKER_TYPE="${P2_WORKER_TYPE:-n2-standard-8}"
NUM_WORKERS="${P2_NUM_WORKERS:-2}"
DISK_GB="${P2_DISK_GB:-100}"
# 2 workers x 8 vCPU: 2 executors per worker with 3 cores each (Lecture 6: 3-5 cores per executor, 2 vCPU reserved)
PROPS="spark.executor.instances=4,spark.executor.cores=3,spark.executor.memory=9g,spark.driver.memory=8g,spark.sql.shuffle.partitions=96,spark.sql.session.timeZone=Asia/Singapore,spark.dynamicAllocation.enabled=false"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BUCKET}/p2/data"
OUT="${BUCKET}/p2/outputs"
RUN_TAG="dataproc-$(date +%Y%m%dT%H%M%S)"

STAGES=(
  "src/ingest.py"
  "checks/ingest_check.py --skip-csv"
  "src/clean.py"
  "checks/clean_ledger.py"
  "checks/coverage_check.py"
  "src/task1.py"
  "src/task2_features.py"
  "src/task2_sample.py"
  "src/task2_train.py"
)

cmd_plan() {
  cat <<EOF
Project ${PROJECT}, region ${REGION}, cluster ${CLUSTER}, image ${IMAGE}
Master ${MASTER_TYPE}; ${NUM_WORKERS} x ${WORKER_TYPE}; ${DISK_GB} GB pd-balanced each.
Rough cost (on-demand list prices, check the console): 20 vCPU of n2 ~ USD 1.0/h + Dataproc fee 0.01/vCPU/h ~ USD 0.2/h
+ disks; the whole run is expected to take about 1 h, so about USD 1.5-3, plus a few cents of storage.
Bucket ${BUCKET}; data ${DATA}; outputs ${OUT}.
Raw data: $([ "${P2_FETCH_ON_CLUSTER:-0}" = "1" ] && echo "downloaded on the cluster (rawfetch)" || echo "uploaded from this machine (bucket)")
Order: plan -> bucket -> cluster -> [rawfetch] -> jobs -> fetch -> delete
Stages: ${STAGES[*]}
EOF
}

cmd_bucket() {
  gcloud storage buckets create "${BUCKET}" --project="${PROJECT}" --location="${REGION}" || true
  if [ "${P2_FETCH_ON_CLUSTER:-0}" != "1" ]; then
    gcloud storage cp "${HERE}"/data/raw/rainfall_*.csv "${DATA}/raw/"
  fi
  (cd "${HERE}" && rm -f /tmp/p2code.zip && zip -qr /tmp/p2code.zip src checks config -x '*/__pycache__/*')
  gcloud storage cp /tmp/p2code.zip "${HERE}/cloud/run_on_cluster.py" "${HERE}/cloud/fetch_raw_on_cluster.py" "${HERE}/cloud/envcheck.py" \
    "${BUCKET}/p2/code/"
}

cmd_cluster() {
  gcloud dataproc clusters create "${CLUSTER}" --project="${PROJECT}" --region="${REGION}" \
    --image-version="${IMAGE}" \
    --master-machine-type="${MASTER_TYPE}" --master-boot-disk-type=pd-balanced --master-boot-disk-size="${DISK_GB}" \
    --num-workers="${NUM_WORKERS}" --worker-machine-type="${WORKER_TYPE}" \
    --worker-boot-disk-type=pd-balanced --worker-boot-disk-size="${DISK_GB}" \
    --public-ip-address --max-idle=30m --enable-component-gateway
  mkdir -p "${HERE}/cloud_run"
  gcloud dataproc clusters describe "${CLUSTER}" --project="${PROJECT}" --region="${REGION}" --format=json \
    > "${HERE}/cloud_run/cluster.json"
}

cmd_rawfetch() {
  gcloud dataproc jobs submit pyspark "${BUCKET}/p2/code/fetch_raw_on_cluster.py" \
    --project="${PROJECT}" --region="${REGION}" --cluster="${CLUSTER}" -- --dest "${DATA}/raw"
}

cmd_jobs() {
  echo "=== envcheck"
  gcloud dataproc jobs submit pyspark "${BUCKET}/p2/code/envcheck.py" \
    --project="${PROJECT}" --region="${REGION}" --cluster="${CLUSTER}"
  for st in "${STAGES[@]}"; do
    script="${st%% *}"; args="${st#"$script"}"
    name="$(basename "${script}" .py)"
    echo "=== ${name}"
    # shellcheck disable=SC2086
    gcloud dataproc jobs submit pyspark "${BUCKET}/p2/code/run_on_cluster.py" \
      --project="${PROJECT}" --region="${REGION}" --cluster="${CLUSTER}" \
      --py-files="${BUCKET}/p2/code/p2code.zip" --properties="${PROPS}" \
      -- --data-root "${DATA}" --out-root "${OUT}" --run-id "${RUN_TAG}-${name}" --stage "${script}" ${args}
  done
}

cmd_fetch() {
  mkdir -p "${HERE}/cloud_run"
  gcloud storage cp -r "${OUT}/outputs" "${OUT}/runs" "${OUT}/models" "${HERE}/cloud_run/"
}

cmd_delete() {
  gcloud dataproc clusters delete "${CLUSTER}" --project="${PROJECT}" --region="${REGION}" --quiet
}

case "${1:-plan}" in
  plan) cmd_plan ;;
  bucket) cmd_bucket ;;
  cluster) cmd_cluster ;;
  rawfetch) cmd_rawfetch ;;
  jobs) cmd_jobs ;;
  fetch) cmd_fetch ;;
  delete) cmd_delete ;;
  *) echo "usage: $0 plan|bucket|cluster|rawfetch|jobs|fetch|delete"; exit 2 ;;
esac
