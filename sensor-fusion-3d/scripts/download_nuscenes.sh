#!/usr/bin/env bash
# nuScenes mini download helper
#
# nuScenes requires a free account + manual download (no direct wget).
# This script prints the steps and verifies the extracted layout.
#
# Usage: bash scripts/download_nuscenes.sh

set -e

DEST="data/v1.0-mini"

echo "============================================"
echo "  nuScenes Mini Download Instructions"
echo "============================================"
echo ""
echo "1. Create a free account at: https://www.nuscenes.org/sign-up"
echo ""
echo "2. Go to: https://www.nuscenes.org/nuscenes#download"
echo "   Download: 'Mini' dataset (~4 GB total)"
echo "   Files needed:"
echo "     - v1.0-mini.tgz          (metadata + annotations)"
echo "     - v1.0-mini_blobs.tgz    (sensor data: images + LiDAR)"
echo ""
echo "3. Extract both archives into: $DEST/"
echo ""
echo "   mkdir -p $DEST"
echo "   tar -xzf v1.0-mini.tgz        -C $DEST"
echo "   tar -xzf v1.0-mini_blobs.tgz  -C $DEST"
echo ""
echo "4. Expected layout after extraction:"
echo "   $DEST/"
echo "     v1.0-mini/"
echo "       attribute.json"
echo "       calibrated_sensor.json"
echo "       ego_pose.json"
echo "       sample.json"
echo "       scene.json"
echo "       ..."
echo "     samples/"
echo "       CAM_FRONT/"
echo "       LIDAR_TOP/"
echo "       ..."
echo "     sweeps/"
echo ""
echo "5. Verify with:"
echo "   python sensor-fusion-3d/src/io/nuscenes_loader.py --dataroot $DEST"
echo ""
echo "============================================"

# If data already exists, verify the layout
if [ -d "$DEST/v1.0-mini" ] && [ -d "$DEST/samples" ]; then
    echo "Found existing nuScenes data at: $DEST"
    echo ""
    echo "Checking layout..."
    SCENES=$(python3 -c "
import json
with open('$DEST/v1.0-mini/scene.json') as f:
    scenes = json.load(f)
print(f'  Scenes     : {len(scenes)}')
with open('$DEST/v1.0-mini/sample.json') as f:
    samples = json.load(f)
print(f'  Samples    : {len(samples)}')
cam_files = list(__import__(\"pathlib\").Path(\"$DEST/samples/CAM_FRONT\").glob(\"*.jpg\"))
lidar_files = list(__import__(\"pathlib\").Path(\"$DEST/samples/LIDAR_TOP\").glob(\"*.pcd.bin\"))
print(f'  CAM_FRONT  : {len(cam_files)} images')
print(f'  LIDAR_TOP  : {len(lidar_files)} scans')
" 2>/dev/null)
    echo "$SCENES"
    echo ""
    echo "Layout looks good. Run the pipeline with:"
    echo "  conda activate sensor-fusion"
    echo "  python sensor-fusion-3d/run.py --config sensor-fusion-3d/configs/nuscenes.yaml"
else
    echo "No data found at $DEST — follow the steps above to download."
fi
