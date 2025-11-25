# openpsytslm
Open-source implementation of the PsyTSLM pipeline enabling synthesis of non-verbal and verbal information during dyadic interactions

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for fast and reliable Python package management. UV provides a unified virtual environment that combines dependencies from both the OpenTSLM and facet submodules.

### Prerequisites

1. Install UV (if not already installed):
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mltrinhammer/openpsytslm.git
   cd openpsytslm
   ```

2. Create and activate a virtual environment with UV:
   ```bash
   # Create virtual environment
   uv venv
   
   # Activate the virtual environment
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   
   # On Windows (Command Prompt):
   .venv\Scripts\activate.bat
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```
0
3. Install all dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```




## Project Structure

- `OpenTSLM/` - Time Series Language Model components
- `facet/` - Facial expression analysis and computer vision modules
- `requirements.txt` - Combined dependencies for the entire project
