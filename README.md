# snowda
Snow Data Assimilation for Hydrological Forecasting

## Installation

Python 3.10 or newer. SUMMA and mizuRoute are compiled executables and are not installed by pip.

```bash
python -m venv "$HOME/venvs/snowda"
source "$HOME/venvs/snowda/bin/activate"
python -m pip install -U pip
python -m pip install -e "/path/to/snowda"
python -m ipykernel install --user --name snowda --display-name "Python (snowda)"
```

Run the kernel command with this environment’s Python. It registers a `Python (snowda)` kernel for Jupyter Notebook, including JupyterHub on a cluster. Optional extras are `gis` (fSCA preprocessing), `download` (snow-data scripts), and `dev` (tests, formatting, and JupyterLab).
