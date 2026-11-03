# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import torch
import einops
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            use_h2_inactive_memory: bool = False,
            h2_max_inactive_age: int = 60,
            h2_recovery_thresh: float = 0.2,
            # CRR-E5 inference-side Sinkhorn arbitration (0 = off):
            sinkhorn_iters: int = 0,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.use_h2_inactive_memory = use_h2_inactive_memory
        self.h2_max_inactive_age = h2_max_inactive_age
        self.h2_recovery_thresh = h2_recovery_thresh
        # CRR-E5: Sinkhorn arbitration over (queries x entities+NEW).  Off by
        # default (iters == 0) -- then the tracker is bit-identical to baseline.
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.num_id_vocabulary = get_model(model).num_id_vocabulary

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        # self.trajectory_features = torch.zeros(())

        self.current_track_results = {}
        # H2 is a strictly causal, inference-only fallback.  Entries are created
        # only when an official MOTIP track expires and are queried only for
        # detections that the unchanged active tracker classified as newborn.
        self.h2_inactive_memory = OrderedDict()
        self.h2_stats = {
            "archived": 0,
            "queried_detections": 0,
            "recovered": 0,
            "expired": 0,
            "evicted_for_birth": 0,
        }
        # Read-only attribution for the dynamic entity-pointer decision rule.
        # These counters do not participate in assignment or tracker state.
        self.pointer_assignment_stats = {
            "frames_with_history": 0,
            "objects_scored": 0,
            "accepted_existing": 0,
            "new_wins": 0,
            "absent_address_wins": 0,
            "existing_below_id_threshold": 0,
            "existing_loses_object_max_conflict": 0,
            "existing_duplicate_after_accept": 0,
        }
        self.pointer_assignment_score_samples = {
            "best_existing": [],
            "new": [],
            "winning": [],
        }
        # Read-only P1 attribution recorder. Gated by env var; when disabled
        # (default) it is inert and the tracker is bit-identical to baseline.
        self.record_attribution = os.environ.get("MOTIP_RECORD_ATTRIBUTION", "0") == "1"
        self.attribution_frames = []
        self._frame_index = 0
        self._last_attribution = None
        self._last_pointer_mentions = None
        return

    @torch.no_grad()
    def update(self, image):
        self._frame_index += 1
        self._h2_age_memory()
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds, query_indices = self._get_activate_detections(
            detr_out=detr_out
        )
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        query_indices = query_indices[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]
        if self.record_attribution:
            self._filter_last_attribution(keep_idxs)

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            query_indices = query_indices[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
            if self.record_attribution:
                self._filter_last_attribution(keep_idxs)
        pass

        # Assign new id labels:
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        # Update the results:
        self.current_track_results = {
            "score": scores.detach().cpu(),
            "category": categories.detach().cpu(),
            # "bbox": boxes * self.bbox_unnorm,
            "bbox": (box_cxcywh_to_xywh(boxes) * self.bbox_unnorm).detach().cpu(),
            "query_index": query_indices.detach().cpu(),
            "output_embed": output_embeds.detach().to(dtype=torch.float16).cpu(),
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
        }

        if self.record_attribution:
            self._record_attribution_frame(id_labels=id_labels, id_pred_labels=id_pred_labels)

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos:
        self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels)

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        pass
        return

    def get_h2_stats(self):
        stats = dict(self.h2_stats)
        stats["remaining"] = len(self.h2_inactive_memory)
        return stats

    def get_pointer_assignment_audit(self):
        stats = dict(self.pointer_assignment_stats)
        stats["tracks_created"] = self.next_id
        stats["id_threshold"] = self.id_thresh
        for name, values in self.pointer_assignment_score_samples.items():
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float32)
            stats[f"{name}_score"] = {
                "mean": tensor.mean().item(),
                "p10": torch.quantile(tensor, 0.10).item(),
                "p50": torch.quantile(tensor, 0.50).item(),
                "p90": torch.quantile(tensor, 0.90).item(),
            }
        return stats

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        query_indices = torch.nonzero(activate_indices, as_tuple=False).flatten()
        # Selecting:
        # logits = logits[activate_indices]
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds, query_indices

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        if self.trajectory_features.shape[0] == 0:
            id_pred_labels = self.num_id_vocabulary * torch.ones(
                boxes.shape[0], dtype=torch.int64, device=boxes.device,
            )
            if self.use_h2_inactive_memory and self.h2_inactive_memory:
                id_pred_labels = self._h2_recover_newborns(
                    boxes=boxes,
                    output_embeds=output_embeds,
                    active_predictions=id_pred_labels,
                )
            return id_pred_labels
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            if self.record_attribution and os.environ.get("MOTIP_RECORD_ATTRIBUTION_V2", "0") == "1":
                pointer = get_model(self.model).entity_pointer
                audit = pointer.pop_last_mention_audit()
                self._last_pointer_mentions = audit[0] if audit is not None else None
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if self.sinkhorn_iters > 0:
                id_logits = self._sinkhorn_normalize(id_logits)
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            if self.use_h2_inactive_memory and self.h2_inactive_memory:
                id_pred_labels = self._h2_recover_newborns(
                    boxes=boxes,
                    output_embeds=output_embeds,
                    active_predictions=id_pred_labels,
                )
            return id_pred_labels

    def _sinkhorn_normalize(self, id_logits: torch.Tensor):
        """CRR-E5 equivariant Sinkhorn arbitration (inference only).

        The [N, V+1] logit matrix (last column = NEW) is augmented with a
        dustbin row so entity columns may dump unmatched mass.  A few
        row/column log-normalisation iterations push the exponentiated matrix
        toward doubly stochastic, injecting the "one mention refers to at most
        one antecedent" competition prior.  Permuting entity columns commutes
        with the normalisation, so equivariance is preserved.  The final row
        normalisation makes the downstream softmax yield the doubly-stochastic
        row.  With sinkhorn_iters == 0 this is never called (baseline path).
        """
        logits = id_logits
        n, k = logits.shape
        dustbin = logits.new_full((1, k), 0.0)
        m = torch.cat([logits, dustbin], dim=0)
        for _ in range(self.sinkhorn_iters):
            m = m - torch.logsumexp(m, dim=1, keepdim=True)
            m = m - torch.logsumexp(m, dim=0, keepdim=True)
        m = m - torch.logsumexp(m, dim=1, keepdim=True)
        return m[:n]

    def _h2_recover_newborns(
            self,
            boxes: torch.Tensor,
            output_embeds: torch.Tensor,
            active_predictions: torch.Tensor,
    ):
        """Second-pass matching against inactive tracks without changing active matches."""
        newborn_idxs = (active_predictions == self.num_id_vocabulary).nonzero(as_tuple=False).flatten()
        if newborn_idxs.numel() == 0 or not self.h2_inactive_memory:
            return active_predictions

        active_label_set = {
            int(label) for label in active_predictions.tolist()
            if int(label) != self.num_id_vocabulary
        }
        # A label can cross the timeout boundary and be matched again on the
        # same frame. Never let a stale memory entry compete with that active ID.
        for label in active_label_set:
            self.h2_inactive_memory.pop(label, None)
        labels = [label for label in self.h2_inactive_memory if label not in active_label_set]
        if not labels:
            return active_predictions
        entries = [self.h2_inactive_memory[label] for label in labels]
        device = output_embeds.device
        features = torch.stack([entry["feature"] for entry in entries], dim=0).to(device=device, dtype=self.dtype)
        memory_boxes = torch.stack([entry["box"] for entry in entries], dim=0).to(device=device, dtype=self.dtype)
        ages = torch.tensor([entry["age"] for entry in entries], dtype=torch.int64, device=device)

        # The pretrained relative-position table has finite support.  Preserve
        # ordering and saturate only the oldest bucket; never fabricate a future token.
        current_time = get_model(self.model).id_decoder.rel_pe_length - 1
        memory_times = (current_time - ages.clamp(min=1, max=current_time)).view(1, -1)
        candidate_labels = torch.tensor(labels, dtype=torch.int64, device=device).view(1, -1)
        seq_info = {
            "trajectory_features": features.view(1, 1, 1, len(labels), -1),
            "trajectory_boxes": memory_boxes.view(1, 1, 1, len(labels), 4),
            "trajectory_id_labels": candidate_labels.view(1, 1, 1, len(labels)),
            "trajectory_times": memory_times.view(1, 1, 1, len(labels)),
            "trajectory_masks": torch.zeros((1, 1, 1, len(labels)), dtype=torch.bool, device=device),
            "unknown_features": output_embeds[newborn_idxs].view(1, 1, 1, -1, output_embeds.shape[-1]),
            "unknown_boxes": boxes[newborn_idxs].view(1, 1, 1, -1, 4),
            "unknown_masks": torch.zeros((1, 1, 1, len(newborn_idxs)), dtype=torch.bool, device=device),
            "unknown_times": torch.full(
                (1, 1, 1, len(newborn_idxs)), current_time, dtype=torch.int64, device=device,
            ),
        }
        seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
        id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
        id_scores = id_logits[0, 0, 0].softmax(dim=-1) if not self.use_sigmoid \
            else id_logits[0, 0, 0].sigmoid()

        # Preserve official object-max semantics, but admit only memory labels and
        # use a dedicated conservative recovery threshold.
        memory_scores = id_scores[:, labels]
        newborn_scores = id_scores[:, self.num_id_vocabulary:self.num_id_vocabulary + 1]
        candidate_scores = torch.cat([memory_scores, newborn_scores], dim=-1)
        best_scores, best_candidate_idxs = torch.max(candidate_scores, dim=-1)
        claimed = set()
        output = active_predictions.clone()
        self.h2_stats["queried_detections"] += len(newborn_idxs)
        order = torch.argsort(best_scores, descending=True)
        memory_label_set = set(labels)
        for local_idx in order.tolist():
            candidate_idx = best_candidate_idxs[local_idx].item()
            label = labels[candidate_idx] if candidate_idx < len(labels) else self.num_id_vocabulary
            score = best_scores[local_idx].item()
            if label in memory_label_set and label not in claimed and score >= self.h2_recovery_thresh:
                output[newborn_idxs[local_idx]] = label
                claimed.add(label)
                self.h2_inactive_memory.pop(label, None)
                self.h2_stats["recovered"] += 1
        return output

    def _h2_age_memory(self):
        if not self.use_h2_inactive_memory:
            return
        expired = []
        for label, entry in self.h2_inactive_memory.items():
            entry["age"] += 1
            if entry["age"] > self.h2_max_inactive_age:
                expired.append(label)
        for label in expired:
            self.h2_inactive_memory.pop(label, None)
            self.h2_stats["expired"] += 1

    def _h2_archive_tracks_expiring_on_cut(self, cut_start: int, current_id_labels: torch.Tensor):
        if not self.use_h2_inactive_memory or cut_start <= 0 or self.trajectory_masks.shape[1] == 0:
            return
        valid = ~self.trajectory_masks
        loses_last_observation = valid[:cut_start].any(dim=0) & ~valid[cut_start:].any(dim=0)
        current_label_set = {int(label) for label in current_id_labels.tolist()}
        for column in loses_last_observation.nonzero(as_tuple=False).flatten().tolist():
            valid_rows = valid[:, column].nonzero(as_tuple=False).flatten()
            if valid_rows.numel() == 0:
                continue
            row = valid_rows[-1].item()
            label = int(self.trajectory_id_labels[row, column].item())
            if label in current_label_set:
                self.h2_inactive_memory.pop(label, None)
                continue
            age = int(self.trajectory_masks.shape[0] - row)
            self.h2_inactive_memory[label] = {
                "feature": self.trajectory_features[row, column].detach().clone(),
                "box": self.trajectory_boxes[row, column].detach().clone(),
                "age": age,
            }
            self.h2_stats["archived"] += 1

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            available = [label for label in self.id_queue if label not in self.h2_inactive_memory]
            while len(available) < n_newborns and self.h2_inactive_memory:
                evict_label = max(
                    self.h2_inactive_memory,
                    key=lambda label: self.h2_inactive_memory[label]["age"],
                )
                self.h2_inactive_memory.pop(evict_label)
                self.id_label_to_id.pop(evict_label, None)
                self.h2_stats["evicted_for_birth"] += 1
                available = [label for label in self.id_queue if label not in self.h2_inactive_memory]
            newborn_id_labels = torch.tensor(
                available[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1], dtype=torch.bool, device=distributed_device(),
            )
            for _ in range(len(newborn_id_labels)):
                if self.trajectory_id_labels.shape[0] > 0:
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                if newborn_id_labels[_].item() in self.id_label_to_id:
                    self.id_label_to_id.pop(newborn_id_labels[_].item())
            # remove from trajectory infos:
            self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
            self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
            self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
            self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
            self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor):
        # 1. cut trajectory infos:
        keep = self.miss_tolerance - 2
        cut_start = max(self.trajectory_features.shape[0] - keep, 0)
        self._h2_archive_tracks_expiring_on_cut(
            cut_start=cut_start,
            current_id_labels=id_labels,
        )
        self.trajectory_features = self.trajectory_features[-keep:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-keep:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-keep:, ...]
        self.trajectory_times = self.trajectory_times[-keep:, ...]
        self.trajectory_masks = self.trajectory_masks[-keep:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            newborn_id_labels = torch.tensor(list(newborn_id_labels), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels)
            _id_labels = einops.repeat(newborn_id_labels, 'n -> t n', t=_T)
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        current_masks[current_idxs] = False
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. a hack implementation to fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        orig_id_scores = id_scores
        if len(id_scores) > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, len(id_scores) - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        for _ in range(len(match_rows)):
            _id = match_cols[_]
            if _id not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            elif _id >= self.num_id_vocabulary:
                id_labels.append(self.num_id_vocabulary)
            elif id_scores[match_rows[_], _id] < self.id_thresh:
                id_labels.append(self.num_id_vocabulary)
            else:
                id_labels.append(_id)
        if self.record_attribution:
            self._last_attribution = self._build_attribution(
                orig_id_scores=orig_id_scores,
                active_labels_sorted=sorted(trajectory_id_labels_set),
                hungarian_labels=list(id_labels),
            )
        return id_labels

    def _filter_last_attribution(self, keep_mask):
        """Keep audit rows aligned after newborn/runtime filtering."""
        if self._last_attribution is None:
            return
        keep = keep_mask.detach().cpu().tolist()
        keys = ("greedy_labels", "hungarian_labels", "best_existing_scores",
                "best_existing_labels", "new_scores", "top2",
                "candidate_scores", "mention_candidates")
        for key in keys:
            values = self._last_attribution.get(key)
            if values is None:
                continue
            if len(values) != len(keep):
                raise RuntimeError(f"P1 alignment mismatch for {key}: {len(values)} != {len(keep)}")
            self._last_attribution[key] = [v for v, flag in zip(values, keep) if flag]

    def _build_attribution(self, orig_id_scores, active_labels_sorted, hungarian_labels):
        """Compact, read-only snapshot of the per-object pointer decision.

        Captures only what offline GT-aligned classification needs: greedy
        argmax (ignoring the global 1-1 constraint), the Hungarian result,
        best-existing score/label, NEW score, and top-2 candidate scores for
        the margin. No mutation of tracker state.
        """
        n = orig_id_scores.shape[0]
        new_col = self.num_id_vocabulary
        if active_labels_sorted:
            active_idx = torch.tensor(active_labels_sorted, device=orig_id_scores.device, dtype=torch.long)
            existing_scores = orig_id_scores.index_select(1, active_idx)            # [N, A]
            best_existing_vals, best_existing_pos = existing_scores.max(dim=-1)     # [N]
            best_existing_labels = [active_labels_sorted[int(p)] for p in best_existing_pos.tolist()]
        else:
            best_existing_vals = orig_id_scores.new_zeros(n)
            best_existing_labels = [new_col] * n
        new_vals = orig_id_scores[:, new_col]
        greedy_labels = [
            best_existing_labels[i] if float(best_existing_vals[i]) >= float(new_vals[i]) else new_col
            for i in range(n)
        ]
        if active_labels_sorted:
            cand_idx = torch.tensor(active_labels_sorted + [new_col], device=orig_id_scores.device, dtype=torch.long)
            cand_scores = orig_id_scores.index_select(1, cand_idx)                  # [N, A+1]
            k = min(2, cand_scores.shape[1])
            top2_vals, top2_pos = cand_scores.topk(k, dim=-1)
            top2 = [
                [{"label": int(cand_idx[int(p)]), "score": float(v)}
                 for v, p in zip(row_vals, row_pos)]
                for row_vals, row_pos in zip(top2_vals.tolist(), top2_pos.tolist())
            ]
        else:
            top2 = [[{"label": new_col, "score": float(new_vals[i].item())}] for i in range(n)]
        candidate_scores = []
        for query_index in range(n):
            row = [{"label": int(label),
                    "score": float(orig_id_scores[query_index, label].item())}
                   for label in active_labels_sorted]
            row.append({"label": new_col, "score": float(new_vals[query_index].item())})
            candidate_scores.append(row)
        mention_rows = self._last_pointer_mentions or [[] for _ in range(n)]
        if len(mention_rows) != n:
            raise RuntimeError(f"P1 mention alignment mismatch: {len(mention_rows)} != {n}")
        mention_candidates = [[{**candidate, "track_id": (
            -1 if int(candidate["label"]) == new_col
            else self.id_label_to_id.get(int(candidate["label"]), -1))}
            for candidate in row] for row in mention_rows]
        return {
            "active_labels": list(active_labels_sorted),
            "candidate_count": len(active_labels_sorted),
            "greedy_labels": greedy_labels,
            "hungarian_labels": list(hungarian_labels),
            "best_existing_scores": best_existing_vals.tolist(),
            "best_existing_labels": best_existing_labels,
            "new_scores": new_vals.tolist(),
            "top2": top2,
            "candidate_scores": candidate_scores,
            "mention_candidates": mention_candidates,
        }

    def _record_attribution_frame(self, id_labels, id_pred_labels):
        res = self.current_track_results
        track_ids = res["id"].tolist()
        boxes = res["bbox"].tolist()           # xywh pixels
        det_scores = res["score"].tolist()
        attr = self._last_attribution
        snap = {int(k): int(v) for k, v in self.id_label_to_id.items()}

        def lab2tid(lab):
            return -1 if int(lab) == self.num_id_vocabulary else snap.get(int(lab), -1)

        objects = []
        for i in range(len(track_ids)):
            obj = {
                "track_id": int(track_ids[i]),
                "box_xywh": [round(float(x), 2) for x in boxes[i]],
                "det_score": float(det_scores[i]),
                "id_label": int(id_labels[i]),
                "id_pred_label": int(id_pred_labels[i]),
            }
            if attr is not None:
                g_lab = attr["greedy_labels"][i]
                h_lab = attr["hungarian_labels"][i]
                obj.update({
                    "greedy_track_id": lab2tid(g_lab),
                    "hungarian_track_id": lab2tid(h_lab),
                    "greedy_is_new": int(g_lab) == self.num_id_vocabulary,
                    "hungarian_is_new": int(h_lab) == self.num_id_vocabulary,
                    "best_existing_score": float(attr["best_existing_scores"][i]),
                    "best_existing_track_id": lab2tid(attr["best_existing_labels"][i]),
                    "new_score": float(attr["new_scores"][i]),
                    "candidate_count": attr["candidate_count"],
                    "top2": [{"is_new": int(t["label"]) == self.num_id_vocabulary,
                              "track_id": lab2tid(t["label"]), "score": t["score"]}
                             for t in attr["top2"][i]],
                    "candidate_scores": [{"is_new": int(t["label"]) == self.num_id_vocabulary,
                              "track_id": lab2tid(t["label"]), "score": t["score"]}
                             for t in attr["candidate_scores"][i]],
                    "mention_candidates": attr["mention_candidates"][i],
                })
            else:
                obj.update({
                    "greedy_track_id": -1, "hungarian_track_id": -1,
                    "greedy_is_new": True, "hungarian_is_new": True,
                    "best_existing_score": 0.0, "best_existing_track_id": -1,
                    "new_score": 0.0, "candidate_count": 0, "top2": [],
                    "candidate_scores": [], "mention_candidates": [],
                })
            objects.append(obj)
        self.attribution_frames.append({
            "frame": int(self._frame_index),
            "active_track_ids": [lab2tid(lab) for lab in (attr["active_labels"] if attr else [])],
            "candidate_count": attr["candidate_count"] if attr else 0,
            "objects": objects,
        })
        self._last_attribution = None

    def get_attribution_records(self):
        return list(self.attribution_frames)

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs
        self.pointer_assignment_stats["frames_with_history"] += 1
        self.pointer_assignment_stats["objects_scored"] += len(id_scores)
        if trajectory_id_labels_set:
            active_labels = torch.tensor(
                sorted(trajectory_id_labels_set), dtype=torch.int64, device=id_scores.device,
            )
            best_existing_scores = id_scores.index_select(1, active_labels).max(dim=-1).values
            self.pointer_assignment_score_samples["best_existing"].extend(best_existing_scores.tolist())
        self.pointer_assignment_score_samples["new"].extend(
            id_scores[:, self.num_id_vocabulary].tolist()
        )
        self.pointer_assignment_score_samples["winning"].extend(object_max_confs.tolist())
        # Get the max confs of each ID label:
        id_max_confs = dict()
        for conf, id_label in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if id_label not in id_max_confs:
                id_max_confs[id_label] = conf
            else:
                # if conf == id_max_confs[id_label]:  # a very rare case
                #     conf = conf - 0.0001
                id_max_confs[id_label] = max(id_max_confs[id_label], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0  # special token

        # Assign ID labels:
        for _ in range(len(object_max_id_labels)):
            if object_max_id_labels[_].item() not in trajectory_id_labels_set:         # not in tracked IDs -> newborn
                if object_max_id_labels[_].item() == self.num_id_vocabulary:
                    self.pointer_assignment_stats["new_wins"] += 1
                else:
                    self.pointer_assignment_stats["absent_address_wins"] += 1
                id_labels.append(self.num_id_vocabulary)
            else:
                _id_label = object_max_id_labels[_].item()
                _conf = object_max_confs[_].item()
                if _conf < self.id_thresh:
                    self.pointer_assignment_stats["existing_below_id_threshold"] += 1
                    id_labels.append(self.num_id_vocabulary)
                elif _conf < id_max_confs[_id_label]:
                    self.pointer_assignment_stats["existing_loses_object_max_conflict"] += 1
                    id_labels.append(self.num_id_vocabulary)
                elif _id_label in id_labels:
                    self.pointer_assignment_stats["existing_duplicate_after_accept"] += 1
                    id_labels.append(self.num_id_vocabulary)
                else:                                                          # normal case
                    self.pointer_assignment_stats["accepted_existing"] += 1
                    id_labels.append(_id_label)

        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)  # final ID labels
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist())   # all tracked ID labels

        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        # Get the max confs of each object:
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:    # a very rare case
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)

        # Assign ID labels:
        for _ in range(len(id_max_obj_idxs)):
            _obj_idx, _id_label, _conf = id_max_obj_idxs[_].item(), _, id_max_confs[_].item()
            if _conf < self.id_thresh or _conf < object_max_confs[_obj_idx]:
                pass
            elif _id_label not in trajectory_id_labels_set:
                pass
            else:
                id_labels[_obj_idx] = _id_label

        return id_labels
