"""
Continued Curriculum Learning for OpenTSLM on NoXi data.

This script loads an already-trained OpenTSLM Flamingo model from HuggingFace
('OpenTSLM/llama-3.2-3b-ecg-flamingo') and performs continued training on
NoXi dyadic interaction data (facial AU time series + transcript descriptions).

It mirrors the CurriculumTrainer from external/opentslm/curriculum_learning.py,
but:
  1. Loads from a HuggingFace checkpoint instead of requiring local stage checkpoints
  2. Uses NoXiCoTQADataset instead of HARCoTQADataset
  3. Has a single training stage ("noxi_cot") rather than the full 5-stage curriculum

Usage:
    python continued_curriculum_learning.py \
        --data_dir results \
        --combined_dir src/descriptions/output/combined \
        --summary_dir src/descriptions/output/summaries \
        --batch_size 2 \
        --num_epochs 30

    # Distributed (multi-GPU):
    torchrun --nproc_per_node=4 continued_curriculum_learning.py \
        --data_dir results \
        --batch_size 2
"""

import sys
import os

# Add our own source tree
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Add project time_series_datasets folder directly to avoid name collision
PROJECT_TS_DATASETS = os.path.join(PROJECT_SRC, "time_series_datasets")
if PROJECT_TS_DATASETS not in sys.path:
    sys.path.insert(0, PROJECT_TS_DATASETS)

# Add opentslm src to path for model + prompt imports
OPENTSLM_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "opentslm", "src")
)
if OPENTSLM_SRC not in sys.path:
    sys.path.insert(0, OPENTSLM_SRC)

import json
import argparse
import datetime
import shutil
from typing import List, Optional, Dict, Any, Callable

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# OpenTSLM model components
from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from model.projector.MLPProjector import MLPProjector
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)

# Our NoXi dataset
from noxiCoTDataset import NoXiCoTQADataset


# ============================================================================
# DEFAULT HF CHECKPOINT
# ============================================================================

DEFAULT_HF_CHECKPOINT = "OpenTSLM/llama-3.2-3b-ecg-flamingo"


# ============================================================================
# CONTINUED CURRICULUM TRAINER
# ============================================================================


class ContinuedCurriculumTrainer:
    """
    Continued curriculum learning trainer for OpenTSLM Flamingo on NoXi data.

    Loads a pre-trained OpenTSLM Flamingo model from HuggingFace and performs
    continued training on NoXi facial AU time series.

    This mirrors the structure of CurriculumTrainer from
    external/opentslm/curriculum_learning.py but is simplified for a single
    continued-training stage.
    """

    def __init__(
        self,
        hf_checkpoint: str = DEFAULT_HF_CHECKPOINT,
        device: str = None,
        gradient_checkpointing: bool = False,
        dist_url: str = "env://",
        dist_backend: str = "nccl",
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0)),
        results_dir: str = None,
    ):
        """
        Initialize the trainer.

        Args:
            hf_checkpoint: HuggingFace model ID for the pre-trained OpenTSLM checkpoint
            device: Device to use ('cuda', 'mps', 'cpu')
            gradient_checkpointing: Enable gradient checkpointing for memory savings
            dist_url: URL for distributed training
            dist_backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
            local_rank: Local GPU rank
            results_dir: Directory for saving checkpoints and results
        """
        self.hf_checkpoint = hf_checkpoint
        self.device = device or self._get_device()
        self.gradient_checkpointing = gradient_checkpointing
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.local_rank = local_rank

        # Distributed training state
        self.rank = 0
        self.world_size = 1
        if self._should_use_distributed():
            self._init_distributed()

        # Initialize model from HF checkpoint
        if self.rank == 0:
            print(f"Loading pre-trained model from: {hf_checkpoint}")
            print(f"Device: {self.device}")
        self.model = self._initialize_model_from_hf()

        # Results directory
        safe_name = hf_checkpoint.split("/")[-1].replace(".", "_").replace("-", "_")
        self.results_dir = results_dir or os.path.join(
            "results", "continued_training", safe_name
        )
        self._create_results_dir()

    # ------------------------------------------------------------------
    # Device / distributed helpers (mirrored from original)
    # ------------------------------------------------------------------

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _should_use_distributed(self) -> bool:
        return ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1) or (
            "LOCAL_RANK" in os.environ and int(os.environ.get("LOCAL_RANK", -1)) >= 0
            and "WORLD_SIZE" in os.environ
        )

    def _init_distributed(self):
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(hours=999),
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        if self.rank == 0:
            print(f"Initialized distributed training with {self.world_size} GPUs")

    def _get_model(self):
        """Get underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Model initialization from HuggingFace
    # ------------------------------------------------------------------

    def _initialize_model_from_hf(self):
        """Initialize OpenTSLMFlamingo and load weights from HuggingFace checkpoint.

        The HF checkpoint 'OpenTSLM/llama-3.2-3b-ecg-flamingo' stores a full
        model state dict. We:
        1. Create the OpenTSLMFlamingo architecture (with the same LLM base)
        2. Download the checkpoint from HF Hub
        3. Load the state dict into the model
        """
        # Determine the base LLM ID from the checkpoint name
        # The ecg-flamingo checkpoint is built on meta-llama/Llama-3.2-3B
        llm_id = "meta-llama/Llama-3.2-3B"

        if self.rank == 0:
            print(f"Creating OpenTSLMFlamingo with base LLM: {llm_id}")

        model = OpenTSLMFlamingo(
            cross_attn_every_n_layers=1,
            gradient_checkpointing=self.gradient_checkpointing,
            llm_id=llm_id,
            device=self.device,
        ).to(self.device)

        # Load the HF checkpoint weights
        if self.rank == 0:
            print(f"Downloading/loading checkpoint: {self.hf_checkpoint}")

        try:
            from huggingface_hub import hf_hub_download
            import glob

            # Try to download model weights from HF Hub
            # The checkpoint may be stored as a single .pt/.bin file or as
            # safetensors shards
            try:
                # Try single-file checkpoint first
                checkpoint_path = hf_hub_download(
                    repo_id=self.hf_checkpoint,
                    filename="best_model.pt",
                    cache_dir=None,
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )

                if "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                else:
                    state_dict = checkpoint

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if self.rank == 0:
                    if missing:
                        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
                    if unexpected:
                        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                    print("Successfully loaded checkpoint from HuggingFace")

            except Exception as e1:
                if self.rank == 0:
                    print(f"Could not load best_model.pt: {e1}")
                    print("Trying safetensors / pytorch_model.bin ...")

                try:
                    # Try safetensors
                    from safetensors.torch import load_file
                    checkpoint_path = hf_hub_download(
                        repo_id=self.hf_checkpoint,
                        filename="model.safetensors",
                        cache_dir=None,
                    )
                    state_dict = load_file(checkpoint_path)
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    if self.rank == 0:
                        if missing:
                            print(f"Missing keys ({len(missing)}): {missing[:5]}...")
                        if unexpected:
                            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                        print("Successfully loaded safetensors checkpoint from HuggingFace")

                except Exception as e2:
                    try:
                        # Try pytorch_model.bin
                        checkpoint_path = hf_hub_download(
                            repo_id=self.hf_checkpoint,
                            filename="pytorch_model.bin",
                            cache_dir=None,
                        )
                        state_dict = torch.load(
                            checkpoint_path, map_location="cpu", weights_only=False
                        )
                        missing, unexpected = model.load_state_dict(state_dict, strict=False)
                        if self.rank == 0:
                            if missing:
                                print(f"Missing keys ({len(missing)}): {missing[:5]}...")
                            if unexpected:
                                print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                            print("Successfully loaded pytorch_model.bin from HuggingFace")

                    except Exception as e3:
                        if self.rank == 0:
                            print(f"Warning: Could not load HF checkpoint weights.")
                            print(f"  best_model.pt: {e1}")
                            print(f"  model.safetensors: {e2}")
                            print(f"  pytorch_model.bin: {e3}")
                            print("Proceeding with freshly initialized model weights.")
                            print("The model architecture is initialized from the "
                                  "base LLM and will need training from scratch.")

        except ImportError:
            if self.rank == 0:
                print("huggingface_hub not installed. Install with: pip install huggingface-hub")
                print("Proceeding with freshly initialized model (no pretrained weights).")

        # Wrap with DDP if distributed
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )
            if self.rank == 0:
                print(f"Wrapped model with DDP for distributed training")

        return model

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _create_results_dir(self):
        os.makedirs(self.results_dir, exist_ok=True)
        stage_dir = os.path.join(self.results_dir, "noxi_cot")
        os.makedirs(stage_dir, exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(stage_dir, "results"), exist_ok=True)

    # ------------------------------------------------------------------
    # Optimizer (mirrored from original Flamingo path)
    # ------------------------------------------------------------------

    def _get_optimizer(self, lr_base: float = 2e-4):
        """Get optimizer with parameter groups for the Flamingo model."""
        model = self._get_model()

        params_to_optimize = [
            (n, p) for n, p in model.named_parameters()
            if p.requires_grad and not getattr(p, "exclude_from_optimizer", False)
        ]

        params_with_wd, params_without_wd = [], []
        for n, p in params_to_optimize:
            if "gated_cross_attn" in n:
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        if self.rank == 0:
            total_trainable = sum(p.numel() for _, p in params_to_optimize)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {total_trainable:,} / {total_params:,} "
                  f"({100.0 * total_trainable / total_params:.2f}%)")
            print(f"Base learning rate: {lr_base:.2e}")

        return AdamW(
            [
                {"params": params_with_wd, "weight_decay": 0.1},
                {"params": params_without_wd, "weight_decay": 0.0},
            ],
            lr=lr_base,
        )

    # ------------------------------------------------------------------
    # DataLoader creation (mirrored from original)
    # ------------------------------------------------------------------

    def _make_data_loader(
        self,
        datasets: List[Dataset],
        shuffle: bool,
        batch_size: int,
        patch_size: int = PATCH_SIZE,
        distribute: bool = False,
    ) -> DataLoader:
        merged_ds = ConcatDataset(datasets)

        if distribute and dist.is_initialized():
            sampler = DistributedSampler(
                merged_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            return DataLoader(
                merged_ds,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )
        else:
            return DataLoader(
                merged_ds,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )

    # ------------------------------------------------------------------
    # Checkpointing (mirrored from original)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, optimizer, scheduler):
        if dist.is_initialized() and self.rank != 0:
            return

        checkpoint_dir = os.path.join(self.results_dir, "noxi_cot", "checkpoints")
        model = self._get_model()

        model_state = model.state_dict()
        if hasattr(self.model, "module"):
            model_state = {
                k.replace("module.", ""): v for k, v in model_state.items()
            }

        checkpoint = {
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "epoch": epoch,
            "hf_checkpoint": self.hf_checkpoint,
        }

        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        # Check disk space
        total, used, free = shutil.disk_usage(checkpoint_dir)
        free_gb = free / (1024 ** 3)
        if self.rank == 0:
            print(f"Disk space: {free_gb:.2f} GB free in {checkpoint_dir}")

        try:
            torch.save(checkpoint, checkpoint_path)
            if self.rank == 0:
                print(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            if self.rank == 0:
                print(f"Failed to save checkpoint: {e}")
            raise

    def _load_checkpoint(self, optimizer, scheduler):
        """Load checkpoint for resuming training.

        Returns:
            (epoch, val_loss) or (None, inf)
        """
        checkpoint_path = os.path.join(
            self.results_dir, "noxi_cot", "checkpoints", "best_model.pt"
        )

        if not os.path.exists(checkpoint_path):
            return None, float("inf")

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        model = self._get_model()

        model_state = checkpoint["model_state"]
        if hasattr(self.model, "module"):
            model_state = {f"module.{k}": v for k, v in model_state.items()}

        try:
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            if missing and self.rank == 0:
                print(f"Warning: {len(missing)} missing keys when loading checkpoint")
            if unexpected and self.rank == 0:
                print(f"Warning: {len(unexpected)} unexpected keys when loading checkpoint")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint.get("epoch", 0), checkpoint.get("val_loss", float("inf"))

    # ------------------------------------------------------------------
    # Loss history (mirrored from original)
    # ------------------------------------------------------------------

    def _save_loss_history(self, epoch: int, train_loss: float, val_loss: float):
        if dist.is_initialized() and self.rank != 0:
            return

        checkpoint_dir = os.path.join(self.results_dir, "noxi_cot", "checkpoints")
        loss_file = os.path.join(checkpoint_dir, "loss_history.txt")

        if not os.path.exists(loss_file):
            with open(loss_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")

        with open(loss_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    def _display_loss_history(self):
        if dist.is_initialized() and self.rank != 0:
            return

        loss_file = os.path.join(
            self.results_dir, "noxi_cot", "checkpoints", "loss_history.txt"
        )
        if not os.path.exists(loss_file):
            return

        try:
            with open(loss_file, "r") as f:
                lines = f.readlines()
            if len(lines) > 2:
                print("Previous loss history:")
                print("   Epoch\tTrain_Loss\tVal_Loss")
                print("   " + "-" * 30)
                start_idx = max(2, len(lines) - 5)
                for line in lines[start_idx:]:
                    if line.strip() and not line.startswith("-"):
                        parts = line.strip().split("\t")
                        if len(parts) == 3:
                            print(f"   {parts[0]}\t{parts[1]}\t{parts[2]}")
                if len(lines) > 7:
                    print(f"   ... and {len(lines) - 7} more epochs")
                print()
        except Exception as e:
            print(f"Could not read loss history: {e}")

    # ------------------------------------------------------------------
    # Evaluation (mirrored from original)
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        test_loader: DataLoader,
        epoch: int = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on the test set."""
        self.model.eval()
        max_new_tokens = 2000

        results_dir_stage = os.path.join(self.results_dir, "noxi_cot", "results")
        results_file = os.path.join(
            results_dir_stage,
            f"test_predictions_rank_{self.rank if dist.is_initialized() else 0}.jsonl",
        )
        final_results_file = os.path.join(results_dir_stage, "test_predictions.jsonl")

        os.makedirs(results_dir_stage, exist_ok=True)

        results = []
        with open(results_file, "w", encoding="utf-8") as fp:
            with torch.no_grad():
                for batch in tqdm(
                    test_loader, desc="Evaluating", disable=self.rank != 0
                ):
                    predictions = self._get_model().generate(
                        batch, max_new_tokens=max_new_tokens
                    )

                    for sample, pred in zip(batch, predictions):
                        result = {
                            "pre_prompt": sample["pre_prompt"],
                            "time_series_text": sample["time_series_text"],
                            "post_prompt": sample["post_prompt"],
                            "generated": pred,
                            "gold": sample["answer"],
                            "session_id": sample.get("session_id", ""),
                            "speaker_id": sample.get("speaker_id", ""),
                            "turn_index": sample.get("turn_index", -1),
                        }
                        results.append(result)
                        fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fp.flush()

        # Synchronize
        if dist.is_initialized():
            dist.barrier()

        # Merge per-rank files on rank 0
        if (not dist.is_initialized()) or self.rank == 0:
            with open(final_results_file, "w", encoding="utf-8") as merged:
                num_ranks = self.world_size if dist.is_initialized() else 1
                for r in range(num_ranks):
                    part = os.path.join(
                        results_dir_stage, f"test_predictions_rank_{r}.jsonl"
                    )
                    if os.path.exists(part):
                        with open(part, "r", encoding="utf-8") as pf:
                            for line in pf:
                                merged.write(line)
            print(f"Test predictions saved to: {final_results_file}")

        metrics = {"test_loss": float("nan")}
        if epoch is not None:
            metrics["epoch"] = epoch

        # Save metrics
        if (not dist.is_initialized()) or self.rank == 0:
            metrics_file = os.path.join(results_dir_stage, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_file}")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"   {k}: {v:.4f}")
                else:
                    print(f"   {k}: {v}")

        if dist.is_initialized():
            dist.barrier()

        return metrics

    # ------------------------------------------------------------------
    # Main training method
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int = 30,
        batch_size: int = None,
        lr_base: float = 2e-4,
        eval_only: bool = False,
        patch_size: int = PATCH_SIZE,
    ) -> Dict[str, Any]:
        """Run continued training on NoXi data.

        Args:
            num_epochs: Maximum number of training epochs
            batch_size: Batch size per GPU (default: from model_config)
            lr_base: Base learning rate for AdamW
            eval_only: Skip training, only evaluate
            patch_size: Patch size for time series padding

        Returns:
            Dict of evaluation metrics
        """
        if batch_size is None:
            batch_size = BATCH_SIZE

        if self.rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Continued Training on NoXi Data")
            print(f"{'=' * 60}")
            print(f"   HF Checkpoint: {self.hf_checkpoint}")
            print(f"   Epochs: {num_epochs}")
            print(f"   Base LR: {lr_base:.2e}")
            print(f"   Batch size per GPU: {batch_size}")
            if self.world_size > 1:
                print(f"   Effective batch size: {batch_size * self.world_size}")
            print(f"   Early stopping patience: {EARLY_STOP_PAT}")
            print(f"   Gradient clip norm: {GRAD_CLIP_NORM}")
            if eval_only:
                print(f"   MODE: Evaluation only")
            print()

        # Initialize optimizer
        optimizer = self._get_optimizer(lr_base)

        # Create data loaders
        eos_token = self._get_model().get_eos_token()

        train_dataset = NoXiCoTQADataset("train", EOS_TOKEN=eos_token)
        val_dataset = NoXiCoTQADataset("validation", EOS_TOKEN=eos_token)
        test_dataset = NoXiCoTQADataset("test", EOS_TOKEN=eos_token)

        if self.rank == 0:
            print(f"Dataset sizes: train={len(train_dataset)}, "
                  f"val={len(val_dataset)}, test={len(test_dataset)}")

        train_loader = self._make_data_loader(
            [train_dataset],
            shuffle=True,
            batch_size=batch_size,
            patch_size=patch_size,
            distribute=self.world_size > 1,
        )
        val_loader = self._make_data_loader(
            [val_dataset],
            shuffle=False,
            batch_size=1,
            patch_size=patch_size,
            distribute=False,
        )
        test_loader = self._make_data_loader(
            [test_dataset],
            shuffle=False,
            batch_size=1,
            patch_size=patch_size,
            distribute=self.world_size > 1,
        )

        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if self.rank == 0:
            print(f"Total training steps: {total_steps}")
            print(f"Warmup steps: {warmup_steps}")

        # Resume from checkpoint if available
        best_epoch, best_val_loss = self._load_checkpoint(optimizer, scheduler)
        if best_epoch is not None:
            if self.rank == 0:
                print(f"Resuming from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
                self._display_loss_history()
        else:
            if self.rank == 0:
                print("Starting continued training from HF checkpoint")
            best_val_loss = float("inf")

        # Skip training if eval_only
        if eval_only:
            if self.rank == 0:
                print("Skipping training (eval_only mode)")
            metrics = self._evaluate(test_loader, epoch=best_epoch)
            return metrics

        # Training loop
        epochs_no_improve = 0
        start_epoch = (best_epoch + 1) if best_epoch is not None else 1

        for epoch in range(start_epoch, num_epochs + 1):
            # Set epoch for distributed sampler
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Training
            self.model.train()
            running_loss = 0.0
            prog = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{num_epochs}",
                disable=self.rank != 0,
            )

            for i, batch in enumerate(prog):
                # Debug: first batch of first epoch
                if epoch == start_epoch and i == 0 and self.rank == 0:
                    print(f"[DEBUG] Batch size: {len(batch)}")
                    if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
                        for k, v in batch[0].items():
                            if hasattr(v, "shape"):
                                print(f"[DEBUG] '{k}' shape: {v.shape}")
                            elif isinstance(v, list):
                                print(f"[DEBUG] '{k}' list length: {len(v)}")

                optimizer.zero_grad()
                loss = self._get_model().compute_loss(batch)
                loss.backward()

                clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                if self.rank == 0:
                    prog.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

            avg_train_loss = running_loss / len(train_loader)
            if self.rank == 0:
                tqdm.write(f"Epoch {epoch} - train loss: {avg_train_loss:.4f}")

            # Validation
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    val_loader, desc="Validating", disable=self.rank != 0
                ):
                    val_loss += self._get_model().compute_loss(batch).item()

            avg_val_loss = val_loss / max(len(val_loader), 1)

            # Synchronize val loss across ranks
            if dist.is_initialized():
                val_tensor = torch.tensor(avg_val_loss, device=self.device)
                dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
                avg_val_loss = val_tensor.item() / self.world_size

            if self.rank == 0:
                tqdm.write(f"Epoch {epoch} - val loss: {avg_val_loss:.4f}")
                tqdm.write(f"Epoch {epoch} - best loss: {best_val_loss:.4f}")

            # Save loss history
            self._save_loss_history(epoch, avg_train_loss, avg_val_loss)

            # Early stopping
            should_save = avg_val_loss + 1e-4 < best_val_loss
            if dist.is_initialized():
                save_tensor = torch.tensor(
                    1 if should_save else 0, device=self.device
                )
                dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM)
                should_save = save_tensor.item() > 0

            if should_save:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                self._save_checkpoint(epoch, avg_val_loss, optimizer, scheduler)
                if self.rank == 0:
                    tqdm.write("New best model saved.\n")
            else:
                epochs_no_improve += 1
                if self.rank == 0:
                    tqdm.write(
                        f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n"
                    )
                if epochs_no_improve >= EARLY_STOP_PAT:
                    if self.rank == 0:
                        tqdm.write(
                            f"\nEarly stopping triggered after {epoch} epochs."
                        )
                    break

            # Synchronize early stopping state
            if dist.is_initialized():
                best_tensor = torch.tensor(best_val_loss, device=self.device)
                ep_tensor = torch.tensor(epochs_no_improve, device=self.device)
                dist.broadcast(best_tensor, src=0)
                dist.broadcast(ep_tensor, src=0)
                best_val_loss = best_tensor.item()
                epochs_no_improve = int(ep_tensor.item())

        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(optimizer, scheduler)
        if best_epoch is not None and self.rank == 0:
            print(f"Loaded best checkpoint from epoch {best_epoch} for evaluation.")

        if self.rank == 0:
            print(f"\nTraining completed.")
            print(f"   Best validation loss: {best_val_loss:.4f}")

        metrics = self._evaluate(test_loader, epoch=best_epoch)

        # Mark as completed
        if (not dist.is_initialized()) or self.rank == 0:
            metrics["completed"] = True
            metrics_file = os.path.join(
                self.results_dir, "noxi_cot", "results", "metrics.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

        return metrics


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Continued Curriculum Learning for OpenTSLM on NoXi data"
    )

    # Model
    parser.add_argument(
        "--hf_checkpoint",
        type=str,
        default=DEFAULT_HF_CHECKPOINT,
        help=f"HuggingFace model checkpoint ID (default: {DEFAULT_HF_CHECKPOINT})",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="results",
        help="Root directory containing NoXi session subdirectories",
    )
    parser.add_argument(
        "--combined_dir",
        type=str,
        default=None,
        help="Directory with session_XXX_combined.json files",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Directory with session_XXX.summary.json files",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to NoXi_MetaData.xlsx for language-based filtering",
    )
    parser.add_argument(
        "--allowed_languages",
        nargs="+",
        default=None,
        help="Languages to include (default: French, German, English)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per split (for debugging)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Max time series sequence length",
    )

    # Training
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=f"Batch size (default: {BATCH_SIZE} from model_config)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Base learning rate"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Skip training, only evaluate from checkpoint",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory for saving results and checkpoints",
    )

    # Device & distributed
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--dist_url", type=str, default="env://", help="Distributed training URL"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="nccl", help="Distributed backend"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank",
    )

    args = parser.parse_args()

    # Configure the dataset class with data paths before creating any instances
    NoXiCoTQADataset.configure(
        data_dir=args.data_dir,
        combined_dir=args.combined_dir,
        summary_dir=args.summary_dir,
        metadata_path=args.metadata_path,
        allowed_languages=args.allowed_languages,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
    )

    # Initialize trainer
    trainer = ContinuedCurriculumTrainer(
        hf_checkpoint=args.hf_checkpoint,
        device=args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        local_rank=args.local_rank,
        results_dir=args.results_dir,
    )

    # Run training
    metrics = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr_base=args.lr,
        eval_only=args.eval_only,
    )

    # Print summary
    print(f"\n{'=' * 40}")
    print("Final Results:")
    print(f"{'=' * 40}")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
