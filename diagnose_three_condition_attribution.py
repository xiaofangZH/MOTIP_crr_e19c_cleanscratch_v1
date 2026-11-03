#!/usr/bin/env python3
"""Three-condition attribution diagnostic for the CRR causal entity pointer.

    A  GT-matched detection tokens   + GT-correct history      -> representation
    B  full runtime detection set    + GT-correct history      -> set impurity
    C  full runtime detection set    + self-predicted history  -> closed loop

GT is used for diagnosis only.  Nothing here is wired into the runtime.

The runtime contract is reproduced literally:
  * assignment protocol object-max (models/runtime_tracker.py:_object_max_assignment)
  * ID_THRESH 0.2 gate on the winning existing score
  * NEWBORN_THRESH 0.6 filter on detections predicted NEW
  * miss_tolerance 30 -> the trajectory grid keeps the last 28 rows and gains the
    current row, i.e. a 29-frame causal window whose absence rows are masked
  * columns whose 28 kept rows are all masked are removed (track expiry)
  * ID labels 0..49 recycled from the free pool in ascending order
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from models.motip.entity_pointer import CausalEntityPointer
from models.motip.trajectory_modeling import TrajectoryModeling

NEW = 50
FEAT_DIM = 256
HIST_ROWS = 28          # miss_tolerance - 2
MATCH_IOU = 0.5
ID_THRESH = 0.2
NEWBORN_THRESH = 0.6


# --------------------------------------------------------------------------- #
# Geometry / GT / observations
# --------------------------------------------------------------------------- #
def xywh_iou(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    return inter / np.maximum(union, 1e-8)


def gt_path(root, sequence):
    root = Path(root)
    for candidate in (root / f"{sequence}.txt", root / "gt.txt",
                      root / sequence / "gt" / "gt.txt"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No GT for {sequence} under {root}")


def load_gt(path):
    by_frame = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) < 6:
                continue
            by_frame[int(fields[0])].append((int(fields[1]), [float(x) for x in fields[2:6]]))
    return by_frame


def match_frame(det_boxes, gt_entries, threshold=MATCH_IOU):
    if len(det_boxes) == 0 or not gt_entries:
        return []
    ious = xywh_iou(det_boxes, [entry[1] for entry in gt_entries])
    rows, cols = linear_sum_assignment(1.0 - ious)
    return [(int(r), int(gt_entries[int(c)][0]), float(ious[r, c]))
            for r, c in zip(rows, cols) if ious[r, c] >= threshold]


def load_observations(path, gt):
    """frame -> list of {feature, box, score, gt_id|None} in detector order."""
    obs = torch.load(path, map_location="cpu")
    frames = obs["frames"].numpy()
    boxes = obs["boxes_xywh"].float().numpy()
    scores = obs["scores"].float().numpy()
    embeds = obs["query_embeddings_fp16"].float()
    detections = {}
    for frame in range(1, int(frames.max()) + 1):
        idxs = np.flatnonzero(frames == frame)
        entries = [{
            "feature": embeds[int(i)],
            "box": torch.tensor(boxes[int(i)]),
            "score": float(scores[int(i)]),
            "gt_id": None,
        } for i in idxs]
        for local_idx, gt_id, _ in match_frame(boxes[idxs], gt.get(frame, [])):
            entries[local_idx]["gt_id"] = gt_id
        detections[frame] = entries
    return detections, int(frames.max())


# --------------------------------------------------------------------------- #
# Modules
# --------------------------------------------------------------------------- #
def load_modules(checkpoint, device):
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw)
    trajectory = TrajectoryModeling(256, 2, 256, False)
    pointer = CausalEntityPointer(
        feature_dim=256,
        num_id_vocabulary=50,
        aggregation_mode="equivariant_memory",
        entity_state_residual_scale=2.0,
        entity_state_hidden_dim=64,
        id_memory_dim=128,
        new_head_context=False,
    )
    trajectory.load_state_dict(
        {k.split("trajectory_modeling.", 1)[1]: v for k, v in state.items()
         if k.startswith("trajectory_modeling.")}, strict=True)
    pointer.load_state_dict(
        {k.split("entity_pointer.", 1)[1]: v for k, v in state.items()
         if k.startswith("entity_pointer.")}, strict=True)
    return trajectory.eval().to(device), pointer.eval().to(device)


@torch.no_grad()
def forward_pointer(trajectory, pointer, hist, current, device):
    """hist = (features [T,N,256], boxes [T,N,4], labels [T,N], masks [T,N])."""
    features, boxes, labels, masks = hist
    t_count, n_slots = labels.shape
    seq = {
        "trajectory_features": features[None, None].to(device),
        "trajectory_boxes": boxes[None, None].to(device),
        "trajectory_id_labels": labels[None, None].to(device),
        "trajectory_times": torch.arange(t_count, device=device).view(1, 1, t_count, 1)
                            .expand(1, 1, t_count, n_slots).contiguous(),
        "trajectory_masks": masks[None, None].to(device),
        "unknown_features": torch.stack([d["feature"] for d in current]).view(
            1, 1, 1, -1, FEAT_DIM).to(device),
        "unknown_boxes": torch.stack([d["box"] for d in current]).view(
            1, 1, 1, -1, 4).to(device),
        "unknown_masks": torch.zeros((1, 1, 1, len(current)), dtype=torch.bool, device=device),
        "unknown_times": torch.full((1, 1, 1, len(current)), t_count,
                                    dtype=torch.long, device=device),
    }
    seq = trajectory(seq)
    logits, _, _ = pointer(seq)
    return logits[0, 0, 0].softmax(dim=-1)


# --------------------------------------------------------------------------- #
# Runtime decision rules (verbatim port of runtime_tracker.py)
# --------------------------------------------------------------------------- #
def object_max_assignment(probs, active_labels, id_thresh=ID_THRESH):
    labels = []
    scores, cols = torch.max(probs, dim=-1)
    id_max = {}
    for conf, col in zip(scores.tolist(), cols.tolist()):
        id_max[col] = max(id_max.get(col, conf), conf)
    if NEW in id_max:
        id_max[NEW] = 0.0
    for conf, col in zip(scores.tolist(), cols.tolist()):
        if col not in active_labels:
            labels.append(NEW)
        elif conf < id_thresh:
            labels.append(NEW)
        elif conf < id_max[col]:
            labels.append(NEW)
        elif col in labels:
            labels.append(NEW)
        else:
            labels.append(col)
    return labels


def newborn_keep(pred, scores, newborn_thresh=NEWBORN_THRESH):
    return {i for i, (label, score) in enumerate(zip(pred, scores))
            if label != NEW or score > newborn_thresh}


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
class Recorder:
    """Rows hold GT-matched detections only: (frame, gt_id, track, birth, forced, active)."""

    def __init__(self):
        self.rows = []
        self.born = 0
        self.matched_tracks = set()
        self.gt_with_detection = set()
        self.counters = Counter()

    def new_track(self):
        self.born += 1
        return self.born

    def add(self, frame, gt_id, track, birth, forced_new, gt_active):
        self.rows.append((frame, int(gt_id), int(track), bool(birth),
                          bool(forced_new), bool(gt_active)))
        if track >= 0:
            self.matched_tracks.add(int(track))

    def summary(self, gt):
        kept = [row for row in self.rows if row[2] >= 0]
        per_gt = defaultdict(list)
        for row in kept:
            per_gt[row[1]].append(row)
        purity, switches, distinct = 0, 0, []
        for rows in per_gt.values():
            tracks = [row[2] for row in rows]
            counts = Counter(tracks)
            purity += counts.most_common(1)[0][1]
            switches += sum(1 for a, b in zip(tracks, tracks[1:]) if a != b)
            distinct.append(len(counts))
        gt_instances = sum(len(v) for v in gt.values())
        result = {
            "gt_instances": gt_instances,
            "gt_with_detection": len(self.gt_with_detection),
            "detector_recall": len(self.gt_with_detection) / max(gt_instances, 1),
            "matched_scored": len(self.rows),
            "kept": len(kept),
            "dropped_by_newborn_filter": len(self.rows) - len(kept),
            "births_on_matched": sum(1 for row in self.rows if row[3]),
            "false_new_while_gt_active": sum(
                1 for row in self.rows if row[3] and not row[4]),
            "weighted_purity": purity / max(len(kept), 1),
            "id_switches": switches,
            "id_switches_per_1000": 1000.0 * switches / max(len(kept), 1),
            "distinct_tracks_per_gt": float(np.mean(distinct)) if distinct else 0.0,
            "gt_count": len(per_gt),
            "tracks_born": self.born,
            "spurious_birth_ratio": 1.0 - len(self.matched_tracks) / max(self.born, 1),
        }
        result.update(self.counters)
        samples = result.pop("prob_samples", 0)
        result["mean_new_prob"] = result.pop("new_prob_sum", 0.0) / max(samples, 1)
        result["mean_best_existing_prob"] = (
            result.pop("best_existing_sum", 0.0) / max(samples, 1))
        existing = result.pop("existing_samples", 0)
        result["argmax_existing_top1"] = (
            result.pop("existing_correct", 0) / max(existing, 1))
        result["mean_correct_col_prob"] = (
            result.pop("correct_col_prob_sum", 0.0) / max(existing, 1))
        return result


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
@torch.no_grad()
def replay_teacher(trajectory, pointer, detections, gt, max_frame, device, use_all):
    """Conditions A (use_all=False) and B (use_all=True)."""
    recorder = Recorder()
    history = defaultdict(list)          # gt_id -> [(frame, feature, box)]
    gt_track = {}
    for frame in range(1, max_frame + 1):
        first = max(1, frame - HIST_ROWS)
        last_seen = {}
        for gt_id, rows in history.items():
            recent = max((r[0] for r in rows if r[0] >= first), default=None)
            if recent is not None:
                last_seen[gt_id] = recent
        active = sorted(last_seen, key=lambda g: (-last_seen[g], g))[:50]
        active_set = set(active)
        for gt_id in list(gt_track):
            if gt_id not in active_set:
                gt_track.pop(gt_id)

        frame_dets = detections.get(frame, [])
        for det in frame_dets:
            if det["gt_id"] is not None:
                recorder.gt_with_detection.add((frame, det["gt_id"]))
        current = frame_dets if use_all else [d for d in frame_dets if d["gt_id"] is not None]
        if not current:
            continue

        if active:
            t_count = frame - first
            n_slots = len(active)
            features = torch.zeros((t_count, n_slots, FEAT_DIM))
            boxes = torch.zeros((t_count, n_slots, 4))
            labels = torch.empty((t_count, n_slots), dtype=torch.long)
            masks = torch.ones((t_count, n_slots), dtype=torch.bool)
            for column, gt_id in enumerate(active):
                labels[:, column] = column
                for time, feature, box in history[gt_id]:
                    if time < first:
                        continue
                    row = time - first
                    features[row, column] = feature
                    boxes[row, column] = box
                    masks[row, column] = False
            probs = forward_pointer(trajectory, pointer,
                                    (features, boxes, labels, masks), current, device)
            pred = object_max_assignment(probs, set(range(n_slots)))
            argmax = probs.argmax(dim=-1).tolist()
        else:
            pred = [NEW] * len(current)
            argmax = [NEW] * len(current)

        column_to_gt = {i: gt_id for i, gt_id in enumerate(active)}
        if active:
            recorder.counters["prob_samples"] += len(current)
            recorder.counters["new_prob_sum"] += float(probs[:, NEW].sum())
            recorder.counters["best_existing_sum"] += float(
                probs[:, :n_slots].max(dim=-1).values.sum())
        keep = newborn_keep(pred, [d["score"] for d in current])
        for idx, det in enumerate(current):
            gt_id = det["gt_id"]
            forced = gt_id not in active_set if gt_id is not None else False
            if gt_id is None:
                if idx in keep and pred[idx] == NEW:
                    recorder.new_track()
                continue
            if idx in keep:
                if pred[idx] == NEW:
                    gt_track[gt_id] = recorder.new_track()
                    recorder.add(frame, gt_id, gt_track[gt_id], True, forced, not forced)
                else:
                    owner = column_to_gt[pred[idx]]
                    if owner not in gt_track:
                        gt_track[owner] = recorder.new_track()
                    recorder.add(frame, gt_id, gt_track[owner], False, False, True)
            else:
                recorder.add(frame, gt_id, -1, pred[idx] == NEW, forced, not forced)

            if not forced and pred[idx] != NEW and column_to_gt.get(pred[idx]) != gt_id:
                recorder.counters["cross_gt_assignment"] += 1
            if active and not forced:
                column = active.index(gt_id)
                recorder.counters["argmax_samples"] += 1
                recorder.counters["argmax_correct"] += int(argmax[idx] == column)
                recorder.counters["existing_samples"] += 1
                recorder.counters["existing_correct"] += int(
                    int(probs[idx, :n_slots].argmax()) == column)
                recorder.counters["correct_col_prob_sum"] += float(probs[idx, column])

        for det in frame_dets:
            if det["gt_id"] is not None:
                history[det["gt_id"]].append((frame, det["feature"], det["box"]))

    result = recorder.summary(gt)
    samples = result.pop("argmax_samples", 0)
    correct = result.pop("argmax_correct", 0)
    result["argmax_top1"] = correct / max(samples, 1)
    assignments = result.get("cross_gt_assignment", 0)
    result["cross_gt_assignment"] = assignments
    result["cross_gt_rate"] = assignments / max(result["matched_scored"], 1)
    return result


@torch.no_grad()
def replay_closed_loop(trajectory, pointer, detections, gt, max_frame, device):
    """Condition C: full detection set + self-predicted history."""
    recorder = Recorder()
    columns = []
    features = torch.zeros((0, 0, FEAT_DIM))
    boxes = torch.zeros((0, 0, 4))
    masks = torch.zeros((0, 0), dtype=torch.bool)
    label_to_track = {}

    for frame in range(1, max_frame + 1):
        current = detections.get(frame, [])
        for det in current:
            if det["gt_id"] is not None:
                recorder.gt_with_detection.add((frame, det["gt_id"]))
        if not current:
            continue

        if columns:
            labels = torch.tensor(columns, dtype=torch.long).view(1, -1).repeat(
                max(features.shape[0], 1), 1)
            probs = forward_pointer(trajectory, pointer,
                                    (features, boxes, labels, masks), current, device)
            pred = object_max_assignment(probs, set(columns))
            recorder.counters["prob_samples"] += len(current)
            recorder.counters["new_prob_sum"] += float(probs[:, NEW].sum())
            recorder.counters["best_existing_sum"] += float(
                probs[:, :len(columns)].max(dim=-1).values.sum())
        else:
            pred = [NEW] * len(current)

        keep = newborn_keep(pred, [d["score"] for d in current])
        n_newborn = sum(1 for i in keep if pred[i] == NEW)
        free = [label for label in range(50) if label not in set(columns)]
        if n_newborn > len(free):
            kept = 0
            pruned = set()
            for i in sorted(keep):
                if pred[i] == NEW:
                    if kept >= len(free):
                        continue
                    kept += 1
                pruned.add(i)
            keep = pruned
            n_newborn = kept
        born = free[:n_newborn]
        for label in born:
            label_to_track[label] = recorder.new_track()
            columns.append(label)

        kept_pred = {}
        cursor = 0
        for i in sorted(keep):
            if pred[i] == NEW:
                kept_pred[i] = born[cursor]
                cursor += 1
            else:
                kept_pred[i] = pred[i]

        for idx, det in enumerate(current):
            if det["gt_id"] is None:
                continue
            if idx in kept_pred:
                recorder.add(frame, det["gt_id"], label_to_track[kept_pred[idx]],
                             pred[idx] == NEW, pred[idx] == NEW, False)
            else:
                recorder.add(frame, det["gt_id"], -1, pred[idx] == NEW, pred[idx] == NEW, False)

        address = {label: i for i, label in enumerate(columns)}
        if features.shape[1] != len(columns):
            features = torch.zeros((0, len(columns), FEAT_DIM))
            boxes = torch.zeros((0, len(columns), 4))
            masks = torch.zeros((0, len(columns)), dtype=torch.bool)
        row_features = torch.zeros((len(columns), FEAT_DIM))
        row_boxes = torch.zeros((len(columns), 4))
        row_masks = torch.ones((len(columns),), dtype=torch.bool)
        for idx, label in kept_pred.items():
            slot = address[label]
            row_features[slot] = current[idx]["feature"]
            row_boxes[slot] = current[idx]["box"]
            row_masks[slot] = False
        features = torch.cat([features[-HIST_ROWS:], row_features[None]], dim=0)
        boxes = torch.cat([boxes[-HIST_ROWS:], row_boxes[None]], dim=0)
        masks = torch.cat([masks[-HIST_ROWS:], row_masks[None]], dim=0)

        alive = (~masks).any(dim=0).tolist()
        if not all(alive):
            selector = torch.tensor(alive)
            columns = [label for label, flag in zip(columns, alive) if flag]
            features = features[:, selector]
            boxes = boxes[:, selector]
            masks = masks[:, selector]

    return recorder.summary(gt)


# --------------------------------------------------------------------------- #
def aggregate(per_condition):
    keys = ("gt_instances", "gt_with_detection", "matched_scored", "kept",
            "dropped_by_newborn_filter", "births_on_matched",
            "false_new_while_gt_active", "id_switches", "gt_count",
            "tracks_born", "cross_gt_assignment")
    totals = {}
    for name, block in per_condition.items():
        if not block:
            continue
        item = {k: sum(seq.get(k, 0) for seq in block.values()) for k in keys}
        item["weighted_purity"] = (
            sum(seq["weighted_purity"] * seq["kept"] for seq in block.values())
            / max(item["kept"], 1))
        item["detector_recall"] = item["gt_with_detection"] / max(item["gt_instances"], 1)
        item["id_switches_per_1000"] = 1000.0 * item["id_switches"] / max(item["kept"], 1)
        item["distinct_tracks_per_gt"] = float(np.mean(
            [seq["distinct_tracks_per_gt"] for seq in block.values()]))
        matched_tracks = sum((1.0 - seq["spurious_birth_ratio"]) * seq["tracks_born"]
                             for seq in block.values())
        item["spurious_birth_ratio"] = 1.0 - matched_tracks / max(item["tracks_born"], 1)
        item["cross_gt_rate"] = item["cross_gt_assignment"] / max(item["matched_scored"], 1)
        for key in ("mean_new_prob", "mean_best_existing_prob",
                    "argmax_existing_top1", "mean_correct_col_prob"):
            seqs = [seq[key] for seq in block.values() if key in seq]
            if seqs:
                item[key] = float(np.mean(seqs))
        if "argmax_top1" in block[list(block)[0]]:
            item["argmax_top1"] = float(np.mean(
                [seq["argmax_top1"] for seq in block.values()]))
        totals[name] = item

    gaps, dominant = {}, None
    if all(name in totals for name in "ABC"):
        a, b, c = (totals[name]["weighted_purity"] for name in "ABC")
        gaps = {"representation_gap_A": 1.0 - a,
                "set_gap_A_minus_B": a - b,
                "loop_gap_B_minus_C": b - c}
        dominant = max(gaps, key=gaps.get)
    return {
        "totals": totals,
        "gaps": gaps,
        "dominant_loss": dominant,
        "reading": {
            "representation_gap_A": "1 - purity(A): pointer quality with GT tokens + GT history",
            "set_gap_A_minus_B": "purity(A) - purity(B): cost of the real detection set",
            "loop_gap_B_minus_C": "purity(B) - purity(C): cost of polluted history",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--conditions", default="ABC")
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectory, pointer = load_modules(args.checkpoint, device)
    per_condition = {name: {} for name in args.conditions}
    for sequence in args.sequences:
        gt = load_gt(gt_path(args.gt_dir, sequence))
        detections, max_frame = load_observations(
            Path(args.observations_dir) / f"{sequence}.pt", gt)
        if args.limit_frames:
            max_frame = min(max_frame, args.limit_frames)
            detections = {k: v for k, v in detections.items() if k <= max_frame}
        gt = {frame: rows for frame, rows in gt.items() if frame <= max_frame}
        for name in args.conditions:
            if name == "A":
                result = replay_teacher(trajectory, pointer, detections, gt,
                                        max_frame, device, use_all=False)
            elif name == "B":
                result = replay_teacher(trajectory, pointer, detections, gt,
                                        max_frame, device, use_all=True)
            elif name == "C":
                result = replay_closed_loop(trajectory, pointer, detections, gt,
                                            max_frame, device)
            else:
                raise ValueError(f"Unknown condition {name}")
            per_condition[name][sequence] = result
            print(f"[{name}] {sequence}: {json.dumps(result)}", flush=True)

    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "protocol": "object-max + ID_THRESH 0.2 + NEWBORN_THRESH 0.6, "
                    "29-frame causal window, 50 recycled labels, IoU>=0.5 GT matching",
        "sequences": args.sequences,
        "per_sequence": per_condition,
        "aggregate": aggregate(per_condition),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()