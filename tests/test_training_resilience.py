import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from data.naive_sampler import NaiveSampler
from models.misc import save_checkpoint
from train import train_one_epoch
from utils.nested_tensor import NestedTensor


class NaiveSamplerResumeTest(unittest.TestCase):
    def test_resume_skips_completed_samples_without_changing_epoch_length(self):
        sampler = NaiveSampler(
            data_source=None,
            sample_steps=[0],
            sample_lengths=[1],
            sample_intervals=[1],
            length_per_iteration=None,
        )
        sampler.sample_infos = [0, 1, 2, 3]

        sampler.set_start_index(2)

        self.assertEqual(list(sampler), [2, 3])
        self.assertEqual(len(sampler), 4)


class CheckpointWriteTest(unittest.TestCase):
    def test_failed_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint_latest.pth"
            checkpoint.write_bytes(b"known-good-checkpoint")

            def fail_after_partial_write(_, path):
                Path(path).write_bytes(b"partial-new-checkpoint")
                raise RuntimeError("simulated write failure")

            with patch("models.misc.is_main_process", return_value=True), patch(
                "models.misc.torch.save", side_effect=fail_after_partial_write
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    save_checkpoint(
                        model=nn.Linear(1, 1),
                        path=str(checkpoint),
                        states={},
                        optimizer=None,
                        scheduler=None,
                    )

            self.assertEqual(checkpoint.read_bytes(), b"known-good-checkpoint")


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, frames, part, **kwargs):
        assert part == "detr"
        return {"pred_logits": self.weight.expand(frames.tensors.shape[0], 1)}


class _DummyCriterion:
    weight_dict = {"loss": 1.0}

    def __call__(self, outputs, targets, batch_len):
        return {"loss": outputs["pred_logits"].sum()}, None


class _DummyAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.tensor(0.0)

    def gather(self, tensor):
        return tensor

    def wait_for_everyone(self):
        pass


class _DummyLoader:
    batch_sampler = type("BatchSampler", (), {"batch_size": 1})()

    def __len__(self):
        return 2

    def __iter__(self):
        for _ in range(2):
            yield {
                "images": NestedTensor(
                    tensors=torch.ones((1, 1, 1, 1, 1), dtype=torch.uint8),
                    mask=torch.zeros((1, 1, 1, 1), dtype=torch.bool),
                ),
                "annotations": [[{
                    "bbox": torch.zeros((0, 4)),
                    "category": torch.zeros((0,), dtype=torch.int64),
                }]],
                "metas": None,
            }


class _DummyLogger:
    def metrics(self, **kwargs):
        pass

    def info(self, **kwargs):
        pass


class PeriodicCheckpointTest(unittest.TestCase):
    def test_periodic_checkpoint_records_completed_step_and_epoch_end_resets_position(self):
        model = _DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1])
        states = {"start_epoch": 0, "global_step": 0, "step_in_epoch": 0}
        saved_states = []
        saved_schedulers = []

        def capture_checkpoint(**kwargs):
            saved_states.append(copy.deepcopy(kwargs["states"]))
            saved_schedulers.append(kwargs["scheduler"])

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "train.save_checkpoint", side_effect=capture_checkpoint
        ), patch("train.torch.cuda.synchronize"), patch(
            "train.torch.cuda.max_memory_allocated", return_value=0
        ):
            train_one_epoch(
                accelerator=_DummyAccelerator(),
                logger=_DummyLogger(),
                states=states,
                epoch=0,
                dataloader=_DummyLoader(),
                model=model,
                detr_criterion=_DummyCriterion(),
                id_criterion=type("IdCriterion", (), {"weight": 1.0})(),
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=True,
                lr_warmup_epochs=0,
                lr_warmup_tgt_lr=0.1,
                detr_num_train_frames=1,
                detr_num_checkpoint_frames=0,
                use_decoder_checkpoint=False,
                accumulate_steps=1,
                separate_clip_norm=False,
                outputs_dir=temp_dir,
                save_checkpoint_per_step=2,
            )

        self.assertEqual(saved_states, [{
            "start_epoch": 0,
            "global_step": 2,
            "step_in_epoch": 2,
        }])
        self.assertEqual(saved_schedulers, [scheduler])
        self.assertEqual(states, {
            "start_epoch": 1,
            "global_step": 2,
            "step_in_epoch": 0,
        })


if __name__ == "__main__":
    unittest.main()
