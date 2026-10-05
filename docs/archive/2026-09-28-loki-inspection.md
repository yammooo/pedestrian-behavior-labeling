# LOKI inspection

Historical context, not current instructions. Covers the 2026-09-25 alignment check and 2026-09-28 qualitative review. Current release counts and remaining checks belong in the [LOKI note](../literature/README.md).

### Point-cloud and 3D-box alignment check

The [official format description](https://usa.honda-ri.com/loki) says `label3d_*.txt` is annotated in point-cloud space. Local `pc_*.ply` files are binary little-endian PLY with vertex `x`, `y`, `z`, and intensity. In four matched frames (`scenario_000`: `0000`, `0050`; `scenario_001`: `0000`; `scenario_079`: `0138`), 70 of 73 nearby vehicle/pedestrian boxes contained at least one point when their raw center, dimensions, and yaw were applied directly to the same-frame PLY (`0.15` coordinate-unit tolerance). Mirroring `y`, swapping `x/y`, or shifting `x` by 10 units reduced this to 22, 10, and 20 boxes. For the 40 vehicle boxes in these frames, yaw-rotated boxes contained 11,460 points, versus 6,169 without rotation and 5,865 with yaw offset by 90°. Some boxes have few points, consistent with sparse or occluded returns; these counts are an alignment sanity check, not annotation-quality metrics.

This supports plotting **same-frame PLY `x/y` points and raw `label3d` boxes together in BEV**, with `z` as height and yaw applied in the `x/y` plane. No extra transform was needed in the checked frames. The ego-forward axis, physical units, odometry/map transforms, and RGB projection remain unverified; do not use this check as camera calibration evidence.

## Selected clip observations (2026-09-28)

These seven tracks were chosen as informative examples. The raw behavior value is in the 3D label row: it is present when the selected pedestrian has a 3D box but no 2D box, and absent in 2D-only frames. `Waiting to cross` often looked consistent with someone held back by traffic or signals, but the examples below show limits of the available view and a possible boundary with `Stopped`.

- `scenario_238 / daea6047-f97b-4e9f-9c54-cffaec34fe82`: Only frames `0000`–`0004` (three frames) have labels, all `Waiting to cross` with 3D boxes only. The person is outside the RGB field of view. The 3D box contains 85 and 190 points in the first and third frames, but none in `0002`, where a truck blocks the LiDAR view; the other point clusters are distinct from the truck but do not clearly look like a pedestrian in BEV. This is difficult to judge from these observations alone.
- `scenario_453 / ff3ec68b-63f7-4fe2-8e45-a6fe3fb80fef`: All 18 labeled frames are `Waiting to cross` with no 2D box; the person is outside the RGB field of view. In BEV the person appears still on a broad sidewalk across the opposing lane, without an apparent approach to the road. The reviewer would have called this `Stopped`, but the sidewalk and road layout is hard to determine from LiDAR alone.
- `scenario_461 / b4c4e4f7-3f23-426e-875c-7d9f713413af`: The label changes from `Waiting to cross` at frame `0028` to `Crossing the road` at `0030`, roughly when the person enters the road (possibly one frame off). Six later frames retain a 2D box but lack a 3D box and behavior label.
- `scenario_154 / 89e1b6ac-454d-40ab-b82b-566aa7471f3b`: Eleven frames are labeled `Stopped`. LiDAR returns are sparse and the person is distant, briefly discernible in RGB from behind near or within the road, apparently facing into a truck.
- `scenario_188 / 4a23f315-7436-4876-b222-203af44dab83`: A person seen from behind, apparently standing on a sidewalk, is labeled `Stopped` for 19 frames.
- `scenario_192 / f9a93b0d-563b-4d4d-b593-55d0d187e854`: A person is partly hidden by a metal barrier at the sidewalk–road edge and clearly visible in only a few frames; all 16 3D-labeled frames say `Stopped`.
- `scenario_197 / 0d770d93-315d-4f20-a573-921c475d6c6e`: A person apparently directing traffic in the road has planted feet but moves their arms. All 75 labeled frames say `Stopped`, consistent with the label allowing gestures while the person remains in place.
