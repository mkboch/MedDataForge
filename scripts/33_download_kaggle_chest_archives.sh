#!/usr/bin/env bash
set -u

cd /home/manikm/meddataforge || exit 1
source .venv/bin/activate

mkdir -p data/images_raw_archives/chest/kaggle results/logs results/tables

LOG="results/logs/33_download_kaggle_chest_archives.inner.log"
SUMMARY="results/tables/kaggle_chest_download_summary.txt"

{
  echo "===== KAGGLE CHEST DOWNLOAD START ====="
  date
  echo "PWD=$(pwd)"
  echo "PYTHON=$(which python)"
  python --version
  echo

  echo "===== KAGGLE AUTH CHECK ====="
  if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "Kaggle token found: $HOME/.kaggle/kaggle.json"
  else
    echo "ERROR: Kaggle token not found."
    exit 1
  fi
  echo

  echo "===== DOWNLOAD 1: Chest X-Ray Images (Pneumonia) ====="
  mkdir -p data/images_raw_archives/chest/kaggle/chest_xray_pneumonia
  kaggle datasets download \
    -d paultimothymooney/chest-xray-pneumonia \
    -p data/images_raw_archives/chest/kaggle/chest_xray_pneumonia \
    --force

  echo
  echo "===== DOWNLOAD 2: COVID-19 Radiography Database ====="
  mkdir -p data/images_raw_archives/chest/kaggle/covid19_radiography
  kaggle datasets download \
    -d tawsifurrahman/covid19-radiography-database \
    -p data/images_raw_archives/chest/kaggle/covid19_radiography \
    --force

  echo
  echo "===== DOWNLOAD 3: Indiana U. Chest X-rays ====="
  mkdir -p data/images_raw_archives/chest/kaggle/indiana_chest_xrays
  kaggle datasets download \
    -d raddar/chest-xrays-indiana-university \
    -p data/images_raw_archives/chest/kaggle/indiana_chest_xrays \
    --force

  echo
  echo "===== ARCHIVE FILES ====="
  find data/images_raw_archives/chest/kaggle -maxdepth 3 -type f -printf "%p\t%k KB\n" | sort

  echo
  echo "===== DISK SPACE ====="
  df -h /home/manikm | tail -n 1

  echo
  echo "===== KAGGLE CHEST DOWNLOAD COMPLETE ====="
  date
} 2>&1 | tee "$LOG"

{
  echo "===== KAGGLE CHEST DOWNLOAD SUMMARY ====="
  date
  echo
  echo "Archive files:"
  find data/images_raw_archives/chest/kaggle -maxdepth 3 -type f -printf "%p\t%k KB\n" | sort
  echo
  echo "Disk:"
  df -h /home/manikm | tail -n 1
  echo
  echo "Log:"
  echo "$LOG"
} > "$SUMMARY"

echo
echo "===== FINAL SUMMARY FILE ====="
cat "$SUMMARY"
