#!/bin/bash
# Downloads KITTI raw drive 2011_09_26_drive_0001
# Synced+rectified data, calibration, and tracklets

BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
DATE="2011_09_26"
DRIVE="2011_09_26_drive_0001"
DATA_DIR="data/kitti/raw/${DATE}"

mkdir -p "${DATA_DIR}"

echo "Downloading calibration..."
wget -c "${BASE_URL}/${DATE}_calib.zip" -O "${DATA_DIR}/${DATE}_calib.zip"
unzip -o "${DATA_DIR}/${DATE}_calib.zip" -d "${DATA_DIR}/"

echo "Downloading synced+rectified data..."
wget -c "${BASE_URL}/${DRIVE}/${DRIVE}_sync.zip" -O "${DATA_DIR}/${DRIVE}_sync.zip"
unzip -o "${DATA_DIR}/${DRIVE}_sync.zip" -d "${DATA_DIR}/"

echo "Downloading tracklets..."
wget -c "${BASE_URL}/${DRIVE}/${DRIVE}_tracklets.zip" -O "${DATA_DIR}/${DRIVE}_tracklets.zip"
unzip -o "${DATA_DIR}/${DRIVE}_tracklets.zip" -d "${DATA_DIR}/"

echo "Done. Data at: ${DATA_DIR}"
