# openpsytslm
Open-source implementation of the PsyTSLM pipeline enabling synthesis of non-verbal and verbal information during dyadic interactions

# TODO: 
I am integrating OpenFace 3.0 and OpenTSLM as submodules.
OpenFace 3.0 does not display its submodules in a .gitmodules file; users that clone this present repo, will therefore not get the full dependencies for Open Face 3.0. Need to find a way to ensure future users of this repo does not face that bug

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

