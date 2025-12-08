# openpsytslm
Open-source implementation of the PsyTSLM pipeline enabling synthesis of non-verbal and verbal information during dyadic interactions

# Regarding OpenFace integration
I am integrating OpenFace 3.0 and OpenTSLM as submodules.
OpenFace 3.0 does not display its submodules in a .gitmodules file; users that clone this present repo, will therefore not get the full dependencies for Open Face 3.0. 
I recommend manually creating a .gitmodules file in the OpenFace-3.0 directory and adding _their_ dependencies to it. Will look for more robust fix moving forward.


## Installation

This project uses [UV](https://docs.astral.sh/uv/) Python package management. 

### Prerequisites

1. Install UV:
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LSf https://astral.sh/uv/install.sh | sh
   ```

### Setup

#### Option A — Clone fresh (recommended)

```bash
git clone --recurse-submodules https://github.com/mltrinhammer/openpsytslm.git
cd openpsytslm
```

#### Option B — If you already cloned the repo

```bash
git submodule update --init --recursive
```

#### Create environment and sync dependencies

```bash
# Create a unified virtual environment using UV
uv venv

# Install / sync dependencies for the repository
uv sync
```

### Language pipeline (`src/process_data/transcribe.py`)
Create your own `config_language.yaml` file with the following lines. Remember to login to HF and request access to pyannote

```yaml
hf_token: "hf_xxxxxxxxxxxxxxxxxxx"
diary_model: pyannote/speaker-diarization-3.1
whisper_model_size: large
whisper_language: en
both_speakers: true
min_segment_ms: 1500
audio_sample_rate: 16000
audio_channels: 1
whisper_beam_size: 3
whisper_batch_size: 8
whisper_compute_type: "float16"

```
