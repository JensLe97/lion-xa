#!/bin/bash

SRC="/_data/datasets/SemanticKITTI/dataset"
TARG="/_data/datasets/SemanticKITTI/semantic_kitti2nuscenes"
TRG_CONF="./config/target_nuscenes.yaml"
CONF="./config/lidar_transfer.yaml"
MSG="Success!!"

# Generate new datasets from all sequences
cd /workspace/lion-xa/LiDAR_Transfer
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 00 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 01 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 02 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 03 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 04 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 05 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 06 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 07 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 08 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 09 -p "$TARG" -t "$TRG_CONF" &&
./lidar_deform.py -d "$SRC" -c "$CONF" -b -w -s 10 -p "$TARG" -t "$TRG_CONF" || \
    MSG="FAILED!!!"

echo $MSG
