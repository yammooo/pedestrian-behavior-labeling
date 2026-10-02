# LOKI inspection

Run from the repository root with `conda activate pedestrian-behavior` and `ffmpeg` on `PATH`:

```bash
python -m pedestrian_behavior.inspection summary --root data/loki_data
python -m pedestrian_behavior.inspection gallery --root data/loki_data \
  --action "Crossing the road" --offset 0 --limit 8 \
  --output outputs/inspection/loki/crossing-000
```

`summary` counts raw actions, tracks, and missing aligned files. `gallery` creates an HTML index and whole-scenario videos for tracks containing the exact action. Each 5 FPS video shows RGB beside a point-cloud BEV, centered on the selected pedestrian in a fixed 40-coordinate-unit view (+x up, +y right). Yellow outlines mark that pedestrian's available 2D and 3D boxes. Missing boxes are marked separately; the BEV center interpolates across missing 3D labels and holds the nearest known position at either end. A missing point cloud leaves a marked blank BEV. Change `--offset` and `--output` for the next batch; existing non-empty output directories are rejected.

## ROAD-Waymo RGB + BEV

Install the optional `waymo-inspection` dependencies (Pillow, NumPy, PyArrow) in
`pedestrian-behavior`; the inspected `aalto` environment already has them. No
Waymo SDK or TensorFlow is needed. Run where the original Waymo data is accessible:

```bash
python -m pip install -e '.[waymo-inspection]'
python -m pedestrian_behavior.inspection.road_waymo \
  --index /home/user20/road_waymo_mapping/merged_pedestrians_20261002 \
  --action Wait2X --limit 8 --offset 0 \
  --output outputs/inspection/road_waymo/waiting-000
```

Use exact native action names (`Stop`, `Wait2X`, `Xing`, `XingFmLft`, etc.).
Selection is deterministic with `--seed`; `--clip` and `--track` can restrict it.
`--waymo-root` relocates original component paths without editing the index.
Existing nonempty output directories are rejected.

Each 1564×604 MP4 contains **all original FRONT frames**, including frames without
the selected track's ROAD annotation. Playback is 5 FPS to match the LOKI gallery;
Waymo's approximately 10 Hz recordings therefore play at roughly half speed.
RGB keeps its aspect ratio inside the 960×604 panel. The 604×604 BEV shows both
returns from all available LiDAR sensors in a fixed 40 m view, with +x up and +y
right. Points and yaw-rotated boxes use the native same-frame vehicle coordinates.
Only the selected pedestrian's available ROAD 2D box and native 3D footprint are
yellow. Action and location labels remain separate; original Cyclist boxes remain
unchanged and class disagreements are marked.

Only the BEV **view center** is interpolated through missing boxes: known positions
are transformed to world coordinates, interpolated by timestamp, then returned to
the current vehicle frame. The nearest known world position is held at either end.
Tracks without any 3D box use a marked ego-centered view. Missing 2D/3D boxes and
point-cloud components are marked separately; absent boxes are never drawn.
An available cloud with no points inside the focused view is marked explicitly.
Identical repeated ROAD observations become one video observation while retaining
all annotation IDs; conflicting repeats raise an error.

LiDAR decoding follows Waymo's [range-image geometry](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/range_image_utils.py)
and [v2 conversion conventions](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/utils/lidar_utils.py),
including per-pixel TOP sensor ego compensation. No sweep accumulation, inferred
associations, camera-synchronized-box substitution or person segmentation is used.
Component streams must have ordered timestamps; malformed/misaligned data fails
explicitly. This is an inspection reader, not a frozen training-loader contract.
