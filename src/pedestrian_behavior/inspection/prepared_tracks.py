"""Portable frame inspector whose numeric values come exclusively from reloaded tracks."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from pedestrian_behavior.datasets import loki, road_waymo as road
from pedestrian_behavior.inspection.render import BEV_SIZE, BEV_SPAN, bev_pixel, footprint, read_points
from pedestrian_behavior.preparation import load_track, source_times


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def preview(image, points, box2d, box3d, output):
    invalid_box = False
    if box3d is not None:
        try:
            values = [float(box3d[k]) for k in ("pos_x", "pos_y", "dim_x", "dim_y", "yaw")]
            invalid_box = not np.isfinite(values).all() or min(values[2:4]) <= 0
        except (TypeError, ValueError, KeyError):
            invalid_box = True
        if invalid_box:
            box3d = None
    rgb = Image.new("RGB", (640, 403), "#111")
    draw = ImageDraw.Draw(rgb)
    if image is not None:
        fitted = ImageOps.contain(image.convert("RGB"), rgb.size)
        offset = ((rgb.width-fitted.width)//2, (rgb.height-fitted.height)//2)
        rgb.paste(fitted, offset)
        if box2d:
            x1,y1,x2,y2 = box2d
            draw.rectangle((offset[0]+x1*fitted.width, offset[1]+y1*fitted.height,
                            offset[0]+x2*fitted.width, offset[1]+y2*fitted.height), outline="yellow", width=2)
    else:
        draw.text((10, 10), "Missing same-frame RGB", fill="white")
    center = (float(box3d["pos_x"]), float(box3d["pos_y"])) if box3d else (0, 0)
    bev = Image.new("RGB", (BEV_SIZE, BEV_SIZE), "#111")
    if points is not None:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        px = np.rint(BEV_SIZE/2 + (points[:,1]-center[1])*BEV_SIZE/BEV_SPAN).astype(int)
        py = np.rint(BEV_SIZE/2 - (points[:,0]-center[0])*BEV_SIZE/BEV_SPAN).astype(int)
        valid = (px>=0)&(px<BEV_SIZE)&(py>=0)&(py<BEV_SIZE)
        pixels = np.asarray(bev).copy(); pixels[py[valid], px[valid]] = 160
        bev = Image.fromarray(pixels)
    bd = ImageDraw.Draw(bev)
    if box3d:
        corners = [bev_pixel(x,y,center) for x,y in footprint(box3d)]
        bd.line(corners+corners[:1], fill="yellow", width=2)
    bd.rectangle((0,0,604,50), fill="black")
    bd.text((8,5), "Native frame: +x up, +y right; 40 native units wide", fill="white")
    bd.text((8,22), "Same-frame center; no interpolated focus", fill="white")
    if points is None: bd.text((8,38), "Missing same-frame point cloud", fill="yellow")
    elif invalid_box: bd.text((8,38), "Unusable native box; ego-centered view", fill="yellow")
    combined = Image.new("RGB", (1244,604), "#111")
    combined.paste(rgb, (0,100)); combined.paste(bev,(640,0))
    combined.save(output)


def native_previews(reader, metadata, folder):
    """Return one relative preview per selected frame, never nearest-image fill."""
    folder.mkdir(parents=True, exist_ok=False)
    identity = json.loads(metadata["track_locator"])
    keys = set(k for k in metadata["source_frames"] if k is not None)
    images, clouds, boxes2d, boxes3d = {}, {}, {}, {}
    if reader.dataset == "loki":
        scenario = reader.root / identity["scenario"]
        paths = dict(loki.frame_files(scenario))
        for key in keys:
            frame = loki.read_frame(scenario, key, paths[key]); person = frame.pedestrians.get(identity["track_id"])
            if frame.paths["image"]:
                with Image.open(frame.paths["image"]) as raw:
                    images[key] = raw.convert("RGB")
                    if person and person.box:
                        b = person.box
                        boxes2d[key] = [b["left"]/raw.width, b["top"]/raw.height,
                            (b["left"]+b["width"])/raw.width, (b["top"]+b["height"])/raw.height]
            if frame.paths["pointcloud"]:
                clouds[key] = np.asarray(list(read_points(frame.paths["pointcloud"])), dtype=np.float64)[:, :3]
            if person and person.label3d:
                boxes3d[key] = person.label3d
    else:
        scene_id, track = identity["clip"], identity["tube_uid"]
        reader.index_scene(scene_id); scene = reader.scenes[scene_id]; name = scene["segment_context_name"]
        def rows(component):
            return road.component_rows(road.component_path(scene, component, reader.waymo_root), name)
        for row in rows("camera_image"):
            ts = row["key.frame_timestamp_micros"]
            if ts in keys and row["key.camera_name"] == 1:
                with Image.open(BytesIO(row["[CameraImageComponent].image"])) as raw:
                    images[ts] = raw.convert("RGB")
        calibration = {r["key.laser_name"]: r for r in rows("lidar_calibration")}
        pixel_poses = {(r["key.frame_timestamp_micros"],r["key.laser_name"]):r for r in rows("lidar_pose")
                       if r["key.frame_timestamp_micros"] in keys}
        for row in rows("lidar"):
            ts, laser = row["key.frame_timestamp_micros"], row["key.laser_name"]
            if ts not in keys or laser not in calibration: continue
            pixel = pixel_poses.get((ts, laser))
            if laser == 1 and (pixel is None or ts not in reader.poses): continue
            for ret in (1,2):
                prefix = f"[LiDARComponent].range_image_return{ret}."
                if row[prefix+"values"] is None: continue
                frame_pose = np.asarray(reader.poses[ts]["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(4,4) if laser == 1 else None
                points = road.range_points(row[prefix+"values"], row[prefix+"shape"], calibration[laser],
                    pixel["[LiDARPoseComponent].range_image_return1.values"] if laser == 1 else None, frame_pose)
                clouds.setdefault(ts, []).append(points)
        clouds = {ts:np.concatenate(parts) for ts, parts in clouds.items()}
        laser_id = reader._tracks[track]["laser_id"]
        for row in road.annotation_rows(reader.index):
            ts = int(row["frame_timestamp_micros"])
            if row["road_clip_id"] == scene_id and row["road_tube_uid"] == track and ts in keys:
                boxes2d[ts] = json.loads(row["road_box_normalized_json"])
        for ts in keys:
            observation = reader.road[scene_id][track].get(ts)
            if observation is None and (track,ts) in reader.camera and ts in images:
                b = reader.camera[track,ts]; p="[CameraBoxComponent].box."
                x,y,w,h=(b[p+k] for k in ("center.x","center.y","size.x","size.y"))
                im=images[ts]; boxes2d[ts]=[(x-w/2)/im.width,(y-h/2)/im.height,(x+w/2)/im.width,(y+h/2)/im.height]
            b = reader.lidar.get((laser_id,ts))
            if b:
                p="[LiDARBoxComponent].box."
                boxes3d[ts] = {"pos_x":b[p+"center.x"], "pos_y":b[p+"center.y"],
                              "dim_x":b[p+"size.x"], "dim_y":b[p+"size.y"], "yaw":b[p+"heading"]}
    # ponytail: previews are collected per review track; stream them if this grows beyond the sixteen-case pack.
    links = []
    for i, key in enumerate(metadata["source_frames"]):
        filename = f"{i:04d}.jpg"
        preview(images.get(key), clouds.get(key), boxes2d.get(key), boxes3d.get(key), folder/filename)
        links.append(folder.name + "/" + filename)
    return links


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Prepared track inspector</title>
<style>body{font:15px system-ui;background:#171b22;color:#eee;max-width:1300px;margin:24px auto;padding:0 16px}select,input{font:inherit;max-width:100%}canvas{background:#222833;border:1px solid #465165;width:100%;height:auto}img{width:100%}.grid{display:grid;grid-template-columns:1fr 2fr;gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}label{display:block;margin:10px 0}.ok{color:#9de2b4}.fail{color:#ffb595}@media(max-width:750px){.grid{display:block}}</style>
<h1>Native → saved track inspector</h1><p>Values come from reloaded NumPy archives. Gaps remain gaps. Native context and canonical coordinates use separate views.</p>
<label>Case <select id="case"></select></label><p id="reason"></p><p id="checks"></p>
<label>5 Hz slot <input id="slot" type="range" min="0" value="0" step="1"><output id="time"></output></label>
<img id="native" alt="Selected source RGB with native box and same-frame LiDAR BEV"><p id="context"></p>
<div class="grid"><div><h2>Fixed canonical trajectory</h2><canvas id="trajectory" width="400" height="400"></canvas><p>+x forward/up, +y left. Pedestrian: yellow; ego: cyan. Lines break at missing positions.</p></div>
<div><h2>Position, velocity and yaw timelines</h2><canvas id="timeline" width="800" height="480"></canvas><p>Each named quantity has its own vertical scale. Gray line marks the inspected slot; missing values are blank.</p></div></div>
<h2>Selected slot and native supervision</h2><pre id="values"></pre><h2>Track / expected checks</h2><pre id="details"></pre>
<script>
const cases=__DATA__,pick=document.getElementById('case'),slider=document.getElementById('slot');
cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.name;pick.appendChild(o)});
function segments(ctx,values,xy,color){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let previous=false;values.forEach((v,i)=>{if(v===null||v.some?.(x=>x===null)){previous=false;return}let p=xy(v,i);if(previous)ctx.lineTo(...p);else ctx.moveTo(...p);previous=true});ctx.stroke()}
function render(){let c=cases[+pick.value],a=c.arrays,m=c.metadata,i=+slider.value,n=m.source_frames.length;
 document.getElementById('reason').textContent=c.reason||'';
 document.getElementById('checks').textContent=c.checks.length?c.checks.map(x=>`${x.pass?'PASS':'FAIL'} ${x.field}`).join(' | '):'Structural reload checks passed; native labels are not physical-motion oracles.';
 document.getElementById('checks').className=c.checks.every(x=>x.pass)?'ok':'fail';
 slider.max=n-1;document.getElementById('time').textContent=` ${i+1}/${n} | requested ${(m.native_extent_us[0]+i*200000)} µs | source ${c.times[i]??'missing'} | key ${m.source_frames[i]??'missing'}`;
 let image=document.getElementById('native');image.hidden=!c.previews?.length;if(c.previews?.length)image.src=c.previews[i];
 document.getElementById('context').textContent=c.previews?.length?'Only the selected source frame is displayed. Native BEV uses its current box center (ego center when absent).':'Synthetic fixture: no recorded RGB or LiDAR.';
 let canvas=document.getElementById('trajectory'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,400,400);
 let points=[...a.ped_position,...a.ego_position].filter(p=>p.every(v=>v!==null)),xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 if(points.length){let lowX=Math.min(...xs),highX=Math.max(...xs),lowY=Math.min(...ys),highY=Math.max(...ys),span=Math.max(highX-lowX,highY-lowY,1),midX=(lowX+highX)/2,midY=(lowY+highY)/2;
 let xy=p=>[200-(p[1]-midY)/span*340,200-(p[0]-midX)/span*340];
 for(let [key,color] of [['ped_position','#ffe36d'],['ego_position','#6de1ff']]){segments(ctx,a[key],xy,color);a[key].forEach((p,j)=>{if(p.every(x=>x!==null)){let [x,y]=xy(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,j===i?5:1.5,0,2*Math.PI);ctx.fill()}})}
 ctx.fillStyle='#eee';ctx.fillText(`x ${lowX.toFixed(2)}–${highX.toFixed(2)} m; y ${lowY.toFixed(2)}–${highY.toFixed(2)} m`,10,390)}
 canvas=document.getElementById('timeline');ctx=canvas.getContext('2d');ctx.clearRect(0,0,800,480);
 let rows=[['ped_position',0],['ped_position',1],['ped_velocity',0],['ped_velocity',1],['ego_position',0],['ego_position',1],['ego_velocity',0],['ego_velocity',1],['ego_yaw',null]];
 rows.forEach(([key,col],r)=>{let v=a[key].map(x=>col===null?x:x[col]),valid=v.filter(x=>x!==null),lo=Math.min(...valid),hi=Math.max(...valid),span=Math.max(hi-lo,.01),y0=r*52;
 ctx.fillStyle='#eee';ctx.fillText(key+(col===null?'':`[${col}]`)+ (valid.length?` ${lo.toFixed(2)}…${hi.toFixed(2)}`:' missing'),6,y0+14);
 if(valid.length)segments(ctx,v,(x,j)=>[235+j/Math.max(n-1,1)*550,y0+43-(x-lo)/span*35],key.startsWith('ped')?'#ffe36d':'#6de1ff');
 ctx.strokeStyle='#aaa';let x=235+i/Math.max(n-1,1)*550;ctx.beginPath();ctx.moveTo(x,y0);ctx.lineTo(x,y0+48);ctx.stroke()});
 let values={requested_time_us:m.native_extent_us[0]+i*200000,source_time_us:c.times[i],source_frame:m.source_frames[i],availability:m.availability[i],native_metadata:m.native_metadata[i],annotations:m.native_annotations[i],issues:m.issues[i]};
 for(let k in a)values[k]=a[k][i];document.getElementById('values').textContent=JSON.stringify(values,null,2);
 document.getElementById('details').textContent=JSON.stringify({locator:m.track_locator,native_extent_us:m.native_extent_us,native_counts:m.native_counts,anchor_valid:m.anchor_valid,checks:c.checks,expected:c.expected},null,2);
}
pick.onchange=()=>{slider.value=0;render()};slider.oninput=render;if(cases.length)render();
</script></html>'''


def write_inspector(output, cases):
    """cases carry file paths, optional expected quantities and same-frame previews."""
    data = []
    for case in cases:
        metadata, arrays = load_track(case["path"])
        expected = case.get("expected", {})
        checks = []
        for field, value in expected.items():
            actual = arrays.get(field, metadata.get(field))
            if field in arrays:
                # JSON null in expected physical arrays represents missing, not zero.
                answer = np.asarray(value, dtype=arrays[field].dtype)
                passed = np.allclose(actual, answer, atol=1e-9, rtol=1e-9, equal_nan=True)
            else:
                passed = actual == value
            checks.append({"field": field, "pass": bool(passed), "actual": json_safe(actual)})
        data.append({k:v for k,v in case.items() if k != "path"} | {"metadata":metadata,"arrays":json_safe(arrays),
            "expected":expected,"checks":checks,"times":source_times(metadata)})
    text = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    Path(output).write_text(HTML.replace("__DATA__", text))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="Frozen native cases: locator/reason")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path)
    parser.add_argument("--loki-transform", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.collection/"manifest.json").read_text())
    if manifest["status"] != "complete": raise ValueError("Incomplete preparation collection")
    reader = loki.TrackReader(args.input,args.loki_transform) if manifest["dataset"] == "loki" else road.TrackReader(args.input,args.waymo_root)
    frozen = json.loads(args.cases.read_text())
    if isinstance(frozen, dict): frozen = frozen[manifest["dataset"]]
    paths = {}
    for filename in manifest["tracks"]:
        path = args.collection/"tracks"/filename
        metadata,_ = load_track(path); paths[metadata["track_locator"]] = path
    args.output.mkdir(parents=True,exist_ok=False)
    cases = []
    for number, case in enumerate(frozen):
        path = paths[case["locator"]]; metadata,_ = load_track(path)
        previews = native_previews(reader,metadata,args.output/f"case-{number:02d}")
        cases.append({"path":path,"name":case["locator"],"reason":case["reason"],"previews":previews})
    write_inspector(args.output/"index.html",cases)
    (args.output/"cases.json").write_text(json.dumps(frozen,indent=2)+"\n")


if __name__ == "__main__": main()
