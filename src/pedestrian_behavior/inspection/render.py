"""RGB and point-cloud BEV video gallery for LOKI tracks."""

from html import escape
from math import cos, sin
from pathlib import Path
import struct
import subprocess

from pedestrian_behavior.datasets import loki


RGB_SIZE = (960, 604)
BEV_SIZE = 604
BEV_SPAN = 40


def read_points(path: Path):
    """Read the four-float vertices in the inspected LOKI PLY format."""
    with path.open("rb") as source:
        header = []
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"{path}: incomplete PLY header")
            try:
                entry = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}: unsupported PLY header") from error
            header.append(entry)
            if entry == "end_header":
                break
        try:
            vertex = next(i for i, line in enumerate(header) if line.startswith("element vertex "))
            count = int(header[vertex].split()[2])
        except (StopIteration, ValueError, IndexError) as error:
            raise ValueError(f"{path}: unsupported PLY vertex declaration") from error
        properties = []
        for line in header[vertex + 1:]:
            if line.startswith("element ") or line == "end_header":
                break
            properties.append(line)
        if (header[:2] != ["ply", "format binary_little_endian 1.0"]
                or properties != [f"property float {name}" for name in ("x", "y", "z", "intensity")]
                or count < 0):
            raise ValueError(f"{path}: unsupported PLY format; expected little-endian float x/y/z/intensity vertices")
        data = source.read(count * 16)
        if len(data) != count * 16:
            raise ValueError(f"{path}: truncated PLY vertices")
    return struct.iter_unpack("<ffff", data)


def position(label: dict) -> tuple[float, float]:
    return float(label["pos_x"]), float(label["pos_y"])


def focus_centers(frames: list[loki.Frame], track_id: str) -> list[tuple[float, float]]:
    known = [(i, position(frame.pedestrians[track_id].label3d))
             for i, frame in enumerate(frames)
             if track_id in frame.pedestrians and frame.pedestrians[track_id].label3d]
    if not known:
        raise ValueError(f"track {track_id}: no 3D positions")
    centers = []
    next_known = 0
    for i in range(len(frames)):
        while next_known < len(known) - 1 and known[next_known][0] < i:
            next_known += 1
        if next_known == 0:
            centers.append(known[0][1])
        else:
            left_i, left = known[next_known - 1]
            right_i, right = known[next_known]
            fraction = min(1, (i - left_i) / (right_i - left_i))
            centers.append(tuple(a + fraction * (b - a) for a, b in zip(left, right)))
    return centers


def bev_pixel(x: float, y: float, center: tuple[float, float]) -> tuple[float, float]:
    scale = BEV_SIZE / BEV_SPAN
    return (BEV_SIZE / 2 + (y - center[1]) * scale,
            BEV_SIZE / 2 - (x - center[0]) * scale)


def footprint(label: dict) -> list[tuple[float, float]]:
    x, y = position(label)
    dx, dy, yaw = float(label["dim_x"]) / 2, float(label["dim_y"]) / 2, float(label["yaw"])
    return [(x + u * cos(yaw) - v * sin(yaw), y + u * sin(yaw) + v * cos(yaw))
            for u, v in ((-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy))]


def render_track(scenario: Path, track_id: str, output: Path) -> None:
    from PIL import Image, ImageDraw

    frames = list(loki.frames(scenario))
    centers = focus_centers(frames, track_id)
    process = None
    try:
        for frame, center in zip(frames, centers):
            if not frame.paths["image"]:
                raise FileNotFoundError(f"{scenario.name} frame {frame.frame_id}: missing image")
            with Image.open(frame.paths["image"]) as source:
                image = source.convert("RGB").resize(RGB_SIZE)
                scale = RGB_SIZE[0] / source.width
            bev = Image.new("RGB", (BEV_SIZE, BEV_SIZE), "#101010")
            bev_draw = ImageDraw.Draw(bev)
            if frame.paths["pointcloud"]:
                points = []
                for x, y, _z, _intensity in read_points(frame.paths["pointcloud"]):
                    px, py = bev_pixel(x, y, center)
                    if 0 <= px < BEV_SIZE and 0 <= py < BEV_SIZE:
                        points.append((round(px), round(py)))
                bev_draw.point(points, fill="#a0a0a0")
            else:
                bev_draw.text((12, 12), "[missing point cloud]", fill="yellow")
            person = frame.pedestrians.get(track_id)
            box = person.box if person else None
            label3d = person.label3d if person else None
            draw = ImageDraw.Draw(image)
            if box:
                x, y = box["left"] * scale, box["top"] * scale
                draw.rectangle((x, y, x + box["width"] * scale, y + box["height"] * scale), outline="yellow", width=3)
            if label3d:
                corners = [bev_pixel(x, y, center) for x, y in footprint(label3d)]
                bev_draw.line(corners + corners[:1], fill="yellow", width=3)
            else:
                bev_draw.text((12, 30), "[missing 3D box]", fill="yellow")
            action = person.action if person and person.action is not None else "unknown"
            label = f"{frame.scenario}  track {track_id}  frame {frame.frame_id}  action {action}"
            if not box:
                label += "  [missing 2D box]"
            draw.rectangle((0, 0, min(RGB_SIZE[0], len(label) * 7 + 12), 28), fill="black")
            draw.text((6, 7), label, fill="white")
            if process is None:
                process = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                     "-video_size", "1564x604", "-framerate", "5", "-i", "-", "-an",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)],
                    stdin=subprocess.PIPE,
                )
            combined = Image.new("RGB", (1564, 604))
            combined.paste(image, (0, 0))
            combined.paste(bev, (960, 0))
            process.stdin.write(combined.tobytes())
        if process is None:
            raise ValueError(f"{scenario.name}: no frames")
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {scenario.name} track {track_id}")
    except Exception:
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
        output.unlink(missing_ok=True)
        raise


def write_index(output: Path, action: str, matches: list[tuple[str, str]], dataset: str = "LOKI") -> None:
    rows = "\n".join(
        f'<li><a href="{escape(scenario + "_" + track_id + ".mp4", quote=True)}">'
        f'{escape(scenario)} — {escape(track_id)}</a></li>'
        for scenario, track_id in matches
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<title>{escape(dataset)}: {escape(action)}</title><h1>{escape(dataset)}: {escape(action)}</h1>'
        f'<p>{len(matches)} tracks</p><ol>{rows}</ol></html>\n'
    )
