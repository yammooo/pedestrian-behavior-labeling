"""Small point-semantic confusion and aligned BEV diagnostics."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def semantic_confusion(targets, predictions, classes):
    """GT -1 is ignored; prediction == classes is an explicit unmatched error."""
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if (targets.ndim != 1 or predictions.shape != targets.shape
            or not np.issubdtype(targets.dtype, np.integer)
            or not np.issubdtype(predictions.dtype, np.integer)
            or (targets < -1).any() or (targets >= classes).any()
            or (predictions < 0).any() or (predictions > classes).any()):
        raise ValueError("Invalid aligned semantic targets/predictions")
    keep = targets >= 0
    return np.bincount(targets[keep]*(classes+1)+predictions[keep],
                       minlength=classes*(classes+1)).reshape(classes, classes+1)


def semantic_metrics(confusion):
    cm = np.asarray(confusion)
    if (cm.ndim != 2 or cm.shape[1] != cm.shape[0]+1
            or not np.issubdtype(cm.dtype, np.integer) or (cm < 0).any()):
        raise ValueError("Expected nonnegative C x (C+1) confusion counts")
    rows = cm.sum(1)
    cols = cm[:, :-1].sum(0)
    tp = np.diag(cm[:, :-1])
    classes = []
    for i in range(len(rows)):
        union = int(rows[i]+cols[i]-tp[i])
        classes.append({"support": int(rows[i]), "predicted": int(cols[i]),
            "precision": float(tp[i]/cols[i]) if cols[i] else (0. if rows[i] else None),
            "recall": float(tp[i]/rows[i]) if rows[i] else None,
            "iou": float(tp[i]/union) if union else None,
            "f1": float(2*tp[i]/(rows[i]+cols[i])) if rows[i]+cols[i] else None})
    supported = [x for x in classes if x["support"]]
    return {"confusion": cm.tolist(), "classes": classes, "points": int(rows.sum()),
        "accuracy": float(tp.sum()/rows.sum()) if rows.sum() else None,
        "supported_classes": len(supported),
        "supported_macro_iou": float(np.mean([x["iou"] for x in supported])) if supported else None,
        "unmatched_predictions": int(cm[:, -1].sum())}


def semantic_bev(xyz, targets, predictions, names, palette, heading):
    """Three identical TOP point views; display bounds do not restrict scoring."""
    xyz = np.asarray(xyz)
    if xyz.shape != (len(targets), 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid point/label correspondence")
    semantic_confusion(targets, predictions, len(names))
    side = 640
    canvas = Image.new("RGB", (1980, 850), "#11151b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((20, 15), heading, fill="white", font=font)
    xy = np.rint(np.c_[side/2-xyz[:, 1]*side/120, side/2-xyz[:, 0]*side/120]).astype(int)
    inside = ((xy >= 0) & (xy < side)).all(1)
    order = np.argsort(xyz[:, 2], kind="stable")
    order = order[inside[order]]
    labeled = targets >= 0
    colors = np.vstack((palette, [230, 80, 210], [65, 70, 80])).astype(np.uint8)
    gt = np.where(labeled, targets, len(names)+1)
    pred = np.where(labeled, predictions, len(names)+1)
    errors = np.full((len(xyz), 3), [65, 70, 80], dtype=np.uint8)
    errors[labeled] = np.where((targets[labeled] == predictions[labeled])[:, None],
                              [60, 200, 115], [245, 75, 65])
    for i, (title, rgb) in enumerate((("Waymo GT (shared classes)", colors[gt]),
                                     ("Frozen nuScenes D-DITR", colors[pred]),
                                     ("Correct green / incorrect red", errors))):
        panel = np.full((side, side, 3), [21, 26, 34], dtype=np.uint8)
        panel[xy[order, 1], xy[order, 0]] = rgb[order]
        x = 20+i*660
        canvas.paste(Image.fromarray(panel), (x, 80))
        draw.text((x, 52), title, fill="white", font=font)
    for i, (name, color) in enumerate(zip([*names, "unmatched prediction", "excluded/unlabeled GT"], colors)):
        x, y = 20+(i%6)*325, 745+(i//6)*27
        draw.rectangle((x,y+3,x+12,y+15), fill=tuple(color))
        draw.text((x+18,y), name, fill="white", font=font)
    draw.text((20, 817), "Annotated TOP returns only. View: +/-60 m, up ego-forward. Scores include all ranges; gray points are unscored.", fill="white", font=font)
    return canvas
