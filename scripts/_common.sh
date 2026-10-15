# sourced by the other scripts: resolves ROOT and the two interpreters.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TORCH_PY:?set TORCH_PY=/path/to/siso-project-env/bin/python (torch environment, requirements/torch-env.yml)}"
: "${SIONNA_PY:?set SIONNA_PY=/path/to/venv-sionna/bin/python (Sionna/TensorFlow venv, requirements/sionna-venv.txt)}"
cd "$ROOT"
