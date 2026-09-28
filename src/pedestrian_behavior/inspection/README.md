# LOKI inspection

Run from the repository root with `conda activate pedestrian-behavior` and `ffmpeg` on `PATH`:

```bash
python -m pedestrian_behavior.inspection summary --root data/loki_data
python -m pedestrian_behavior.inspection gallery --root data/loki_data \
  --action "Crossing the road" --offset 0 --limit 8 \
  --output outputs/inspection/loki/crossing-000
```

`summary` counts raw actions, tracks, and missing aligned files. `gallery` creates an HTML index and whole-scenario videos for tracks containing the exact action. Each 5 FPS video shows RGB beside a point-cloud BEV, centered on the selected pedestrian in a fixed 40-coordinate-unit view (+x up, +y right). Yellow outlines mark that pedestrian's available 2D and 3D boxes. Missing boxes are marked separately; the BEV center interpolates across missing 3D labels and holds the nearest known position at either end. A missing point cloud leaves a marked blank BEV. Change `--offset` and `--output` for the next batch; existing non-empty output directories are rejected.
