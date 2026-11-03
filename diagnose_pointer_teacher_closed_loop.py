#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from models.motip.entity_pointer import CausalEntityPointer
from models.motip.trajectory_modeling import TrajectoryModeling


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


def load_gt(path):
    by_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            fields = line.strip().split(",")
            if len(fields) < 6:
                continue
            frame, identity = int(fields[0]), int(fields[1])
            box = [float(x) for x in fields[2:6]]
            by_frame[frame].append((identity, box))
    return by_frame


def match_frame(det_boxes, gt_entries, threshold=0.5):
    if len(det_boxes) == 0 or not gt_entries:
        return []
    gt_boxes = [entry[1] for entry in gt_entries]
    ious = xywh_iou(det_boxes, gt_boxes)
    rows, cols = linear_sum_assignment(1.0 - ious)
    return [(int(r), gt_entries[int(c)][0], float(ious[r, c]))
            for r, c in zip(rows, cols) if ious[r, c] >= threshold]


def load_matched_observations(path, gt):
    obs = torch.load(path, map_location="cpu")
    frames = obs["frames"].numpy()
    boxes = obs["boxes_xywh"].float().numpy()
    features = obs["query_embeddings_fp16"].float()
    matched = defaultdict(list)
    for frame in np.unique(frames):
        idxs = np.flatnonzero(frames == frame)
        for local_idx, gt_id, iou in match_frame(boxes[idxs], gt.get(int(frame), [])):
            absolute = int(idxs[local_idx])
            matched[int(frame)].append({
                "gt_id": int(gt_id), "feature": features[absolute],
                "box": torch.tensor(boxes[absolute]), "iou": iou,
            })
    return matched, int(frames.max())


def load_modules(checkpoint, device):
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("model", raw)
    trajectory = TrajectoryModeling(256, 2, 256, False)
    pointer = CausalEntityPointer(256, 50)
    trajectory.load_state_dict({k.split("trajectory_modeling.", 1)[1]: v for k, v in state.items()
                                if k.startswith("trajectory_modeling.")})
    pointer.load_state_dict({k.split("entity_pointer.", 1)[1]: v for k, v in state.items()
                            if k.startswith("entity_pointer.")})
    return trajectory.eval().to(device), pointer.eval().to(device)


def bucket_for_gap(gap):
    if gap == 1:
        return "gap_1"
    if gap <= 5:
        return "gap_2_5"
    if gap <= 15:
        return "gap_6_15"
    return "gap_16_29"


@torch.no_grad()
def teacher_metrics(matched, max_frame, checkpoint, device):
    trajectory, pointer = load_modules(checkpoint, device)
    history = defaultdict(list)
    counts = Counter()
    gaps = defaultdict(Counter)
    margins = []
    correct_probs = []
    candidate_counts = []
    for frame in range(1, max_frame + 1):
        current = matched.get(frame, [])
        if not current:
            continue
        cutoff = frame - 29
        active_ids = sorted(identity for identity, records in history.items()
                            if any(record[0] >= cutoff for record in records))
        address = {identity: idx for idx, identity in enumerate(active_ids)}
        if active_ids:
            first = max(1, cutoff)
            times = list(range(first, frame))
            t_count, n_count = len(times), len(active_ids)
            hist_features = torch.zeros((1, 1, t_count, n_count, 256), device=device)
            hist_boxes = torch.zeros((1, 1, t_count, n_count, 4), device=device)
            hist_masks = torch.ones((1, 1, t_count, n_count), dtype=torch.bool, device=device)
            hist_labels = torch.empty((1, 1, t_count, n_count), dtype=torch.long, device=device)
            hist_times = torch.empty_like(hist_labels)
            for identity, column in address.items():
                hist_labels[..., column] = column
                for row, time in enumerate(times):
                    hist_times[0, 0, row, column] = time
                for time, feature, box in history[identity]:
                    if time < first:
                        continue
                    row = time - first
                    hist_features[0, 0, row, column] = feature.to(device)
                    hist_boxes[0, 0, row, column] = box.to(device)
                    hist_masks[0, 0, row, column] = False
        else:
            # The pointer is undefined without a history candidate; all current
            # observations are necessarily NEW and are recorded directly.
            for item in current:
                counts["total"] += 1
                counts["correct"] += 1
                counts["new_total"] += 1
                counts["new_correct"] += 1
                counts["hungarian_correct"] += 1
            for item in current:
                history[item["gt_id"]].append((frame, item["feature"], item["box"]))
            continue

        current_features = torch.stack([item["feature"] for item in current]).to(device)
        current_boxes = torch.stack([item["box"] for item in current]).to(device)
        seq = {
            "trajectory_features": hist_features,
            "trajectory_boxes": hist_boxes,
            "trajectory_id_labels": hist_labels,
            "trajectory_times": hist_times,
            "trajectory_masks": hist_masks,
            "unknown_features": current_features.view(1, 1, 1, -1, 256),
            "unknown_boxes": current_boxes.view(1, 1, 1, -1, 4),
            "unknown_masks": torch.zeros((1, 1, 1, len(current)), dtype=torch.bool, device=device),
            "unknown_times": torch.full((1, 1, 1, len(current)), frame, dtype=torch.long, device=device),
        }
        seq = trajectory(seq)
        logits, _, _ = pointer(seq)
        logits = logits[0, 0, 0]
        probs = logits.softmax(dim=-1)
        targets = torch.tensor([address.get(item["gt_id"], 50) for item in current], device=device)
        predictions = logits.argmax(dim=-1)
        # Teacher-forced local ranking.
        for idx, item in enumerate(current):
            target = int(targets[idx])
            pred = int(predictions[idx])
            is_new = target == 50
            counts["total"] += 1
            counts["correct"] += int(pred == target)
            counts["new_total" if is_new else "existing_total"] += 1
            counts["new_correct" if is_new and pred == target else
                   "existing_correct" if (not is_new and pred == target) else "incorrect"] += 1
            target_logit = logits[idx, target]
            wrong = logits[idx].clone(); wrong[target] = -1.0e4
            margins.append(float(target_logit - wrong.max()))
            correct_probs.append(float(probs[idx, target]))
            candidate_counts.append(len(active_ids) + 1)
            if not is_new:
                last_seen = max(x[0] for x in history[item["gt_id"]] if x[0] < frame)
                bucket = bucket_for_gap(frame - last_seen)
                gaps[bucket]["total"] += 1
                gaps[bucket]["correct"] += int(pred == target)
        # Teacher-forced global Hungarian accuracy under the runtime protocol.
        candidate_scores = torch.cat([
            probs[:, :len(active_ids)], probs[:, 50:51].repeat(1, max(len(current), 1))
        ], dim=1)
        rows, cols = linear_sum_assignment(1.0 - candidate_scores.cpu().numpy())
        assigned = [50] * len(current)
        for row, col in zip(rows, cols):
            assigned[int(row)] = int(col) if col < len(active_ids) else 50
        counts["hungarian_correct"] += sum(int(pred == int(target)) for pred, target in zip(assigned, targets))
        for item in current:
            history[item["gt_id"]].append((frame, item["feature"], item["box"]))

    def summary(values):
        values = np.asarray(values, dtype=np.float64)
        return {"mean": float(values.mean()), "p10": float(np.quantile(values, .1)),
                "p50": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9))}

    total = counts["total"]
    return {
        "checkpoint": str(checkpoint), "matched_observations": total,
        "top1": counts["correct"] / total,
        "hungarian_accuracy": counts["hungarian_correct"] / total,
        "existing_top1": counts["existing_correct"] / max(counts["existing_total"], 1),
        "new_top1": counts["new_correct"] / max(counts["new_total"], 1),
        "existing_count": counts["existing_total"], "new_count": counts["new_total"],
        "margin": summary(margins), "correct_probability": summary(correct_probs),
        "candidate_count": summary(candidate_counts),
        "gap_accuracy": {name: {"count": x["total"], "accuracy": x["correct"] / x["total"]}
                         for name, x in gaps.items()},
    }


def closed_loop_metrics(tracker_path, gt, max_frame):
    predictions = defaultdict(list)
    with open(tracker_path) as f:
        for line in f:
            fields = line.strip().split(",")
            predictions[int(fields[0])].append((int(fields[1]), [float(x) for x in fields[2:6]]))
    gt_history = defaultdict(list)
    pred_to_gt = defaultdict(Counter)
    switches_by_quarter = [0, 0, 0, 0]
    matches_by_quarter = [0, 0, 0, 0]
    for frame in range(1, max_frame + 1):
        entries = predictions.get(frame, [])
        pairs = match_frame([x[1] for x in entries], gt.get(frame, []))
        quarter = min(3, (frame - 1) * 4 // max_frame)
        for det_idx, gt_id, _ in pairs:
            pred_id = entries[det_idx][0]
            matches_by_quarter[quarter] += 1
            if gt_history[gt_id] and gt_history[gt_id][-1][1] != pred_id:
                switches_by_quarter[quarter] += 1
            gt_history[gt_id].append((frame, pred_id))
            pred_to_gt[pred_id][gt_id] += 1
    total_assignments = sum(sum(x.values()) for x in pred_to_gt.values())
    pure_assignments = sum(max(x.values()) for x in pred_to_gt.values() if x)
    return {
        "matched_assignments": total_assignments,
        "weighted_pred_track_purity": pure_assignments / max(total_assignments, 1),
        "unique_pred_ids_per_gt": {str(gt_id): len(set(pred for _, pred in rows))
                                   for gt_id, rows in gt_history.items()},
        "switches_by_frame_quarter": switches_by_quarter,
        "matches_by_frame_quarter": matches_by_quarter,
        "switch_rate_by_frame_quarter": [s / max(m, 1) for s, m in zip(switches_by_quarter, matches_by_quarter)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--trained-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = load_gt(args.gt)
    matched, max_frame = load_matched_observations(args.observations, gt)
    report = {
        "schema_version": 1,
        "protocol": "IoU>=0.5 detector-to-GT matching; strict prior 29-frame teacher history",
        "sequence": "dancetrack0044", "max_frame": max_frame,
        "teacher_init": teacher_metrics(matched, max_frame, args.init_checkpoint, device),
        "teacher_trained_epoch1": teacher_metrics(matched, max_frame, args.trained_checkpoint, device),
        "closed_loop_hungarian": closed_loop_metrics(args.tracker, gt, max_frame),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
