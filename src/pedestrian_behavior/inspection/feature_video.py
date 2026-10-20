"""Target-centered RGB windows and metric semantic BEV, without model policy."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .feature_probe import relative_xyz, square_crop

VIDEO_SIZE = (1440, 900)
PANEL_SIZE = 640
RING_COLORS = ("#36d9ff", "#ffe247", "#ff68da")
RADII = (15, 20, 30)


def crop_outlines():
    """Pixel edges of nested 1.5/2/3x windows within the 3x display crop."""
    return [(PANEL_SIZE/2*(1-s/3), PANEL_SIZE/2*(1+s/3)) for s in (1.5, 2., 3.)]


def bev_pixels(offsets):
    """64 m square: initial-forward is up, initial-right is right."""
    xy = np.asarray(offsets)[:, :2]
    return PANEL_SIZE/2+np.c_[-xy[:, 1], -xy[:, 0]]*(PANEL_SIZE/64)


def rgb_window(frame, track):
    font = ImageFont.load_default(size=18)
    left = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), "#1c222b")
    ld = ImageDraw.Draw(left)
    image = frame["image"] if frame else None
    box = frame["camera"].get(track) if frame else None
    pixels, crop = square_crop(image, box, expansion=3, output_size=PANEL_SIZE)
    if pixels is not None:
        left = Image.fromarray(np.uint8(np.clip(pixels, 0, 1)*255))
        ld = ImageDraw.Draw(left)
        bounds = np.array(crop["bounds_pixels"])
        b = (np.array(box)*np.tile(image.size, 2)-np.tile(bounds[:2], 2))*PANEL_SIZE/(bounds[2]-bounds[0])
        ld.rectangle(tuple(b), outline="white", width=1)
        for (low, high), color in zip(crop_outlines(), RING_COLORS):
            ld.rectangle((max(1, low), max(1, low), min(PANEL_SIZE-2, high), min(PANEL_SIZE-2, high)), outline=color, width=2)
    else:
        ld.text((30, 280), "No target RGB crop at this timestep", fill="white", font=font)
        ld.text((30, 310), "No box interpolation or previous-frame reuse", fill="#b8c2cc", font=font)
    return left, crop, pixels is not None


def colored_bev(frame, track, anchor, offsets, colors):
    """Shared metric display for semantic and continuous feature colors."""
    font = ImageFont.load_default(size=18)
    right = np.full((PANEL_SIZE, PANEL_SIZE, 3), [21, 26, 34], dtype=np.uint8)
    counts = [0, 0, 0]
    if offsets is not None:
        offsets, colors = np.asarray(offsets), np.asarray(colors)
        if offsets.ndim != 2 or offsets.shape[1] != 3 or colors.shape != (len(offsets), 3) or not np.isfinite(offsets).all() or not np.isfinite(colors).all():
            raise ValueError("Invalid aligned point offsets/colors")
        radius = np.linalg.norm(offsets[:, :2], axis=1)
        counts = [int((radius <= r).sum()) for r in RADII]
        keep = radius <= RADII[-1]
        points, colors = offsets[keep], colors[keep]
        # Draw higher surfaces last where several voxel features share a BEV pixel.
        order = np.argsort(points[:, 2], kind="stable")
        xy = np.rint(bev_pixels(points[order])).astype(int)
        inside = ((xy >= 0) & (xy < PANEL_SIZE)).all(1)
        right[xy[inside, 1], xy[inside, 0]] = colors[order][inside]
    bev = Image.fromarray(right)
    bd = ImageDraw.Draw(bev)
    if offsets is not None:
        for radius, color in zip(RADII, RING_COLORS):
            r = radius*PANEL_SIZE/64
            c = PANEL_SIZE/2
            bd.ellipse((c-r, c-r, c+r, c+r), outline=color, width=2)
            bd.text((c+8, c-r+5), f"{radius} m", fill=color, font=font)
        center, size, yaw = frame["boxes"][track]
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.c_[np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*np.array(size[:2])/2, np.zeros(4)]
        rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        corners = corners @ rotation.T+center
        footprint = bev_pixels(relative_xyz(corners, center, frame["world_from_native"], anchor))
        bd.line([tuple(p) for p in footprint]+[tuple(footprint[0])], fill="white", width=2)
        bd.line((312,320,328,320), fill="white", width=1)
        bd.line((320,312,320,328), fill="white", width=1)
    else:
        bd.text((30, 280), "No usable target 3D box / scan / pose", fill="white", font=font)
        bd.text((30, 310), "No interpolated center or stale features", fill="#b8c2cc", font=font)
    return bev, counts


def render_track_frame(frame, track, anchor, offsets, labels, palette, names, heading):
    canvas = Image.new("RGB", VIDEO_SIZE, "#11151b")
    draw = ImageDraw.Draw(canvas)
    font, title = ImageFont.load_default(size=18), ImageFont.load_default(size=23)
    draw.text((25, 15), heading, fill="white", font=title)
    draw.text((25, 50), "RGB: 3x window; thin outlines show 1.5x / 2x / 3x sides", fill="white", font=font)
    draw.text((760, 50), "LiDAR semantic predictions: 15 / 20 / 30 m radii", fill="white", font=font)
    left, crop, rgb_valid = rgb_window(frame, track)
    canvas.paste(left, (25, 90))
    if offsets is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(offsets),) or (labels < 0).any() or (labels >= len(palette)).any():
            raise ValueError("Invalid aligned semantic labels")
    bev, counts = colored_bev(frame, track, anchor, offsets, palette[labels] if offsets is not None else None)
    canvas.paste(bev, (760, 90))
    for i, (scale, radius, color) in enumerate(zip((1.5,2,3),RADII,RING_COLORS)):
        draw.text((25+i*205, 745), f"{scale:g}x RGB / {radius}m LiDAR", fill=color, font=font)
    draw.text((760, 745), "Up: initial ego forward; right: initial ego right", fill="white", font=font)
    draw.text((25, 775), "White: native target box. RGB displayed at 640px; model input remains 256px.", fill="#b8c2cc", font=font)
    draw.text((760, 775), f"Voxel points <=15/20/30m: {counts[0]}/{counts[1]}/{counts[2]}", fill="#b8c2cc", font=font)
    for i, (name, color) in enumerate(zip(names, palette)):
        x, y = 25+(i%8)*176, 810+(i//8)*27
        draw.rectangle((x,y+3,x+12,y+15), fill=tuple(color))
        draw.text((x+18,y), name.replace("construction_vehicle", "construction"), fill="white", font=ImageFont.load_default(size=14))
    draw.text((25, 872), "Frozen nuScenes D-DITR predictions; no segmentation GT. Whole-track 5 Hz grid; missing observations retained.", fill="#b8c2cc", font=ImageFont.load_default(size=16))
    return canvas, {"crop": crop, "points_within_radii": counts, "rgb_valid": rgb_valid,
                    "lidar_valid": offsets is not None}


PCA_VIDEO_SIZE = (2048, 850)


def render_pca_frame(frame, track, anchor, offsets, colors, heading):
    canvas = Image.new("RGB", PCA_VIDEO_SIZE, "#11151b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    draw.text((25, 12), heading, fill="white", font=font)
    rgb, crop, valid = rgb_window(frame, track)
    for x, title in zip((25, 704, 1383), ("RGB 3x; 1.5x / 2x outlines", "D-DITR: frozen nuScenes features", "Utonia: frozen multi-dataset features")):
        draw.text((x, 53), title, fill="white", font=font)
    canvas.paste(rgb, (25, 90))
    counts = None
    for x, name in ((704, "dditr"), (1383, "utonia")):
        rgb_colors = np.uint8(np.clip(colors[name], 0, 1)*255) if offsets is not None else None
        bev, counts = colored_bev(frame, track, anchor, offsets, rgb_colors)
        canvas.paste(bev, (x, 90))
    draw.text((25, 750), "Same native points; 15 / 20 / 30 m rings. Up: initial ego forward. White: native target box.", fill="white", font=font)
    draw.text((25, 780), "Fixed PCA per encoder across frames/datasets; colors have no class meaning and do not match between encoders.", fill="#b8c2cc", font=font)
    draw.text((25, 811), "Qualitative development probe; Utonia pretraining includes Waymo. No behavior training or accuracy claim.", fill="#b8c2cc", font=font)
    return canvas, {"crop": crop, "rgb_valid": valid, "lidar_valid": offsets is not None,
                    "points_within_radii": counts}
