"""
NoXi Chain-of-Thought QA Dataset for OpenTSLM.

This dataset mirrors the structure of HARCoTQADataset from the original OpenTSLM,
adapting it for NoXi dyadic interaction data with facial Action Unit (AU) time series.

Each sample consists of:
- Multiple AU intensity time series (one per AU), each normalized with mean/std
- A pre-prompt instructing the model to analyze facial AU patterns in a dyadic interaction
- A post-prompt requesting a rationale
- An answer (combined description from combine_transcripts_with_time_series_descriptions_noxi.py,
  or fallback to summaries from summarize_noxi.py)

The dataset class extends QADataset (from opentslm) and implements all required abstract methods.
"""

import sys
import os
import numpy as np
import torch
from typing import List, Tuple, Literal

# Ensure this file's own directory is on sys.path so sibling modules
# (noxiloader, dyadloader, …) can be imported by name without a
# package-qualified path that would collide with opentslm's
# identically-named "time_series_datasets" package.
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Add the opentslm src directory to path so we can import from the original framework
OPENTSLM_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "opentslm", "src")
)
if OPENTSLM_SRC not in sys.path:
    sys.path.insert(0, OPENTSLM_SRC)

from datasets import Dataset as HFDataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset

# Local loader — imported by filename, NOT via the time_series_datasets
# package, to avoid collision with opentslm's package of the same name.
from noxiloader import load_noxi_cot_splits, DEFAULT_AU_COLUMNS


# ============================================================================
# AU LABEL TEMPLATES
# ============================================================================

# Human-readable labels for each AU intensity channel (no facial movement text).
AU_LABELS = {
    "AU04_r": "The following is the facial Action Unit AU04 intensity",
    "AU06_r": "The following is the facial Action Unit AU06 intensity",
    "AU12_r": "The following is the facial Action Unit AU12 intensity",
    "AU15_r": "The following is the facial Action Unit AU15 intensity",
}


# ============================================================================
# DATASET CLASS
# ============================================================================


class NoXiCoTQADataset(QADataset):
    """
    NoXi Chain-of-Thought QA Dataset.

    Mirrors HARCoTQADataset but uses facial AU intensity time series from
    NoXi dyadic interactions instead of accelerometer data.

    Each sample provides:
    - Expert AU time series (17 channels, each normalized)
    - Novice AU time series (17 channels, each normalized)
    - A chain-of-thought answer combining AU pattern descriptions with
      transcript summaries (from combine_transcripts_with_time_series_descriptions_noxi.py)
    """

    # Class-level configuration — can be overridden before instantiation
    data_dir: str = "results"
    combined_dir: str = None
    summary_dir: str = None
    metadata_path: str = None
    allowed_languages: List[str] = None
    train_sessions: List[str] = None
    val_sessions: List[str] = None
    test_sessions: List[str] = None
    au_columns: List[str] = ["AU04_r", "AU06_r", "AU12_r", "AU15_r"]
    max_samples: int = None
    max_seq_length: int = 4096

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    @classmethod
    def configure(
        cls,
        data_dir: str = "results",
        combined_dir: str = None,
        summary_dir: str = None,
        metadata_path: str = None,
        allowed_languages: List[str] = None,
        train_sessions: List[str] = None,
        val_sessions: List[str] = None,
        test_sessions: List[str] = None,
        au_columns: List[str] = None,
        max_samples: int = None,
        max_seq_length: int = 4096,
    ):
        """Configure data paths before instantiation.

        Call this before creating any NoXiCoTQADataset instances to set
        data directories and filtering options.
        """
        cls.data_dir = data_dir
        cls.combined_dir = combined_dir
        cls.summary_dir = summary_dir
        cls.metadata_path = metadata_path
        cls.allowed_languages = allowed_languages
        cls.train_sessions = train_sessions
        cls.val_sessions = val_sessions
        cls.test_sessions = test_sessions
        cls.au_columns = au_columns or ["AU04_r", "AU06_r", "AU12_r", "AU15_r"]
        cls.max_samples = max_samples
        cls.max_seq_length = max_seq_length
        # Reset loaded flag so next instantiation reloads
        if hasattr(cls, "loaded"):
            delattr(cls, "loaded")

    # ------------------------------------------------------------------
    # QADataset abstract methods
    # ------------------------------------------------------------------

    def _load_splits(self) -> Tuple[HFDataset, HFDataset, HFDataset]:
        """Load the NoXi CoT dataset splits using noxiloader.

        Returns:
            Tuple of (train, validation, test) HuggingFace Dataset objects
        """
        train_samples, val_samples, test_samples = load_noxi_cot_splits(
            data_dir=self.__class__.data_dir,
            combined_dir=self.__class__.combined_dir,
            summary_dir=self.__class__.summary_dir,
            metadata_path=self.__class__.metadata_path,
            allowed_languages=self.__class__.allowed_languages,
            train_sessions=self.__class__.train_sessions,
            val_sessions=self.__class__.val_sessions,
            test_sessions=self.__class__.test_sessions,
            au_columns=self.__class__.au_columns,
            max_samples=self.__class__.max_samples,
            max_seq_length=self.__class__.max_seq_length,
        )

        # Convert list-of-dicts to HuggingFace Datasets (same as har_cot_loader)
        train_ds = HFDataset.from_list(train_samples) if train_samples else HFDataset.from_dict({})
        val_ds = HFDataset.from_list(val_samples) if val_samples else HFDataset.from_dict({})
        test_ds = HFDataset.from_list(test_samples) if test_samples else HFDataset.from_dict({})

        print(f"[NoXiCoTQADataset] Split sizes: "
              f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        return train_ds, val_ds, test_ds

    def _get_answer(self, row) -> str:
        """Get the chain-of-thought answer for a sample.

        Uses the combined_description (from combine script) or falls back
        to original_summary.
        """
        answer = row.get("answer", "")
        if not answer:
            answer = row.get("original_summary", "")
        return answer

    def _get_pre_prompt(self, _row) -> str:
        """Get the system/instruction pre-prompt.

        Mirrors the HARCoTQADataset prompt style but adapted for
        facial AU analysis in dyadic interactions.
        """
        speaker = _row.get("speaker_id", "participant")
        text = f"""
        You are given facial Action Unit (AU) intensity time series data from a dyadic interaction (NoXi corpus).
        The data includes AU activations for both the expert and the novice participant during a conversation segment.
        Your task is to analyze the facial behavior patterns and describe what is happening in this interaction.

        Instructions:
        - Begin by analyzing each AU time series without assuming a specific interpretation.
        - Consider the temporal dynamics: are activations sustained, transient, or rhythmic?
        - Think about co-occurrences: which AUs activate together and what facial expressions might they form?
        - Compare the expert's and novice's AU patterns to identify interaction dynamics.
        - Consider the context: this is a {speaker}'s turn in a dyadic conversation.
        - Write your analysis as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
        - Integrate your observations about the AU patterns with any apparent communicative or emotional significance.

        - Make sure to end your response with a concise summary of the interaction dynamics.
        """
        return text

    def _get_post_prompt(self, _row) -> str:
        """Get the post-prompt text."""
        return "Analysis:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Convert AU time series data into TextTimeSeriesPrompt objects.

        This mirrors the HARCoTQADataset pattern exactly:
        1. Each AU channel becomes a separate TextTimeSeriesPrompt
        2. Each time series is normalized (zero-mean, unit-variance)
        3. The text label includes the original mean and std

        We provide both expert and novice AUs as separate channels:
        - Expert AU01_r, Expert AU02_r, ... Expert AU45_r
        - Novice AU01_r, Novice AU02_r, ... Novice AU45_r
        """
        prompts = []
        au_columns = row.get("au_columns") or ["AU04_r", "AU06_r", "AU12_r", "AU15_r"]

        for role, au_key, stats_key in [
            ("expert", "expert_au_vectors", "expert_au_stats"),
            ("novice", "novice_au_vectors", "novice_au_stats"),
        ]:
            au_vectors = row.get(au_key, {})
            au_stats = row.get(stats_key, {})

            for col in au_columns:
                if col not in au_vectors:
                    continue

                raw_signal = au_vectors[col]
                if isinstance(raw_signal, list):
                    signal = torch.tensor(raw_signal, dtype=torch.float32)
                else:
                    signal = torch.as_tensor(raw_signal, dtype=torch.float32)

                if signal.numel() == 0:
                    continue

                # Check for invalid data
                if torch.isnan(signal).any() or torch.isinf(signal).any():
                    print(f"[NoXiCoTQADataset] Invalid data in {role} {col}, "
                          f"session {row.get('session_id', '?')}")
                    continue

                # Normalize (same pattern as HARCoTQADataset)
                mean_val = signal.mean()
                std_val = signal.std()
                min_std = 1e-6
                std_val = torch.clamp(std_val, min=min_std)
                signal_norm = (signal - mean_val) / std_val

                # Check for NaN/Inf after normalization
                if torch.isnan(signal_norm).any() or torch.isinf(signal_norm).any():
                    print(f"[NoXiCoTQADataset] NaN/Inf after normalization for "
                          f"{role} {col}, session {row.get('session_id', '?')}")
                    continue

                # Build text label with mean and std (matches HAR pattern)
                au_label = AU_LABELS.get(col, f"The following is the facial Action Unit {col} intensity")
                text_prompt = (
                    f"{au_label} for the {role}, "
                    f"it has mean {mean_val.item():.4f} and std {std_val.item():.4f}:"
                )

                prompts.append(
                    TextTimeSeriesPrompt(text_prompt, signal_norm.numpy())
                )

        if not prompts:
            # Fallback: provide at least one dummy time series to avoid errors
            print(f"[NoXiCoTQADataset] Warning: no valid AU vectors for "
                  f"session {row.get('session_id', '?')}, turn {row.get('turn_index', '?')}")
            prompts.append(
                TextTimeSeriesPrompt(
                    "No AU data available, it has mean 0.0000 and std 1.0000:",
                    np.zeros(10, dtype=np.float32),
                )
            )

        return prompts

    # ------------------------------------------------------------------
    # Extra fields (like HARCoTQADataset._format_sample)
    # ------------------------------------------------------------------

    def _format_sample(self, row):
        """Format a sample, adding NoXi-specific metadata fields."""
        sample = super()._format_sample(row)

        # Preserve metadata for evaluation / analysis
        sample["session_id"] = row.get("session_id", "")
        sample["speaker_id"] = row.get("speaker_id", "")
        sample["turn_index"] = row.get("turn_index", -1)
        sample["start_ms"] = row.get("start_ms", 0)
        sample["end_ms"] = row.get("end_ms", 0)
        sample["original_summary"] = row.get("original_summary", "")
        sample["original_timeseries_description"] = row.get("original_timeseries_description", "")

        return sample


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=== NoXi CoT QA Dataset Demo ===\n")

    # Configure with defaults
    NoXiCoTQADataset.configure(
        data_dir="results",
        max_samples=5,
    )

    dataset = NoXiCoTQADataset(split="train", EOS_TOKEN="")
    dataset_val = NoXiCoTQADataset(split="validation", EOS_TOKEN="")
    dataset_test = NoXiCoTQADataset(split="test", EOS_TOKEN="")

    print(f"Dataset sizes: Train: {len(dataset)}, "
          f"Validation: {len(dataset_val)}, Test: {len(dataset_test)}")

    if len(dataset) > 0:
        print("\n" + "=" * 50 + "\n")
        print("Sample from training set:")
        sample = dataset[0]
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Session ID: {sample.get('session_id', 'N/A')}")
        print(f"  Speaker: {sample.get('speaker_id', 'N/A')}")
        print(f"  Turn index: {sample.get('turn_index', 'N/A')}")
        print(f"  Time window: {sample.get('start_ms', 'N/A')}ms - {sample.get('end_ms', 'N/A')}ms")

        answer = sample.get("answer", "")
        print(f"  Answer length: {len(answer)}")
        print(f"  Answer preview: {answer[:200]}..." if len(answer) > 200 else f"  Answer: {answer}")

        ts_texts = sample.get("time_series_text", [])
        ts_data = sample.get("time_series", [])
        print(f"  Number of time series channels: {len(ts_texts)}")
        if ts_texts:
            print(f"  First channel text: {ts_texts[0]}")
        if ts_data:
            first_ts = ts_data[0]
            if hasattr(first_ts, '__len__'):
                print(f"  First channel length: {len(first_ts)}")

        print(f"  Pre-prompt: {sample.get('pre_prompt', '')[:200]}...")
        print(f"  Post-prompt: {sample.get('post_prompt', '')}")
