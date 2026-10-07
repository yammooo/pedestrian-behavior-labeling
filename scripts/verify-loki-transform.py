from pathlib import Path
import csv, numpy as np, json
from collections import defaultdict
import argparse
parser = argparse.ArgumentParser(description="Compare stationary native markers under alternative LOKI transforms")
parser.add_argument("--root", type=Path, default=Path("data/loki_data"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.root
results=defaultdict(list)
heading_cosines = []
for scenario in sorted(root.glob('scenario_*')):
    identities=defaultdict(list)
    previous_odometry = None
    for path in sorted(scenario.glob('label3d_*.txt')):
        suffix=path.stem.split('_')[1]
        odom=np.array(list(map(float,(scenario/f'odom_{suffix}.txt').read_text().strip().split(','))))
        if previous_odometry is not None:
            delta = odom[:2] - previous_odometry[:2]
            distance = np.linalg.norm(delta)
            if distance > .02:
                heading = previous_odometry[5]
                heading_cosines.append(float(delta @ [np.cos(heading), np.sin(heading)] / distance))
        previous_odometry = odom
        x,y,z,r,p,h=odom; cr,sr,cp,sp,ch,sh=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(h),np.sin(h)
        rot=np.array([[ch*cp,ch*sp*sr-sh*cr,ch*sp*cr+sh*sr],[sh*cp,sh*sp*sr+ch*cr,sh*sp*cr-ch*sr],[-sp,cp*sr,cp*cr]])
        with path.open() as f:
            for row in csv.DictReader(f,skipinitialspace=True):
                if row['labels'] not in ('Potential_Destination','Road_Entrance_Exit'): continue
                xyz=np.array([float(row['pos_'+a]) for a in 'xyz'])
                identities[(row['labels'],row['track_id'])].append((xyz,odom[:3],rot))
    for (label,tid), obs in identities.items():
        if len(obs)<5: continue
        for mode in ('raw','full','yaw','inverse','y-reflected','swapxy'):
            pts=[]
            for xyz,pos,rot in obs:
                if mode=='raw': v=xyz
                elif mode=='full': v=rot@xyz+pos
                elif mode=='yaw':
                    h=np.arctan2(rot[1,0],rot[0,0]); c,s=np.cos(h),np.sin(h)
                    v=np.array([[c,-s,0],[s,c,0],[0,0,1]])@xyz+pos
                elif mode=='inverse': v=rot.T@xyz+pos
                elif mode=='y-reflected': v=rot@(xyz*np.array([1,-1,1]))+pos
                else: v=rot@xyz[[1,0,2]]+pos
                pts.append(v)
            pts=np.array(pts)
            results[(label,mode)].append(float(np.sqrt(np.mean(np.sum((pts-pts.mean(axis=0))**2,axis=1)))))
summary = {str(k):{'tracks':len(v),'median_rms':float(np.median(v)),'p90_rms':float(np.quantile(v,.9))} for k,v in results.items()}
summary["ego-heading-motion-cosine"] = {"intervals_over_2cm": len(heading_cosines),
    "median": float(np.median(heading_cosines)), "p10": float(np.quantile(heading_cosines, .1)),
    "qualification": "Supports forward axis; reverse/stationary/turning intervals are not filtered by labels"}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
