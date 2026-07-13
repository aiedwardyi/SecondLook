# Setup (Mac)

From a clean machine to the running app. No prior Python setup assumed.

## 1. Install Python 3.12

Option A - python.org installer: https://www.python.org/downloads/

Option B - Homebrew:

```
brew install python@3.12
```

Confirm in a new terminal:

```
python3 --version
```

You want `Python 3.12.x`.

## 2. Clone the repo

```
git clone https://github.com/aiedwardyi/SecondLook.git
cd SecondLook
```

If you already have the folder, `cd` into it instead.

## 3. Create a virtual environment and install deps

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

`requirements.txt` pins CUDA wheels of PyTorch for Windows/Linux NVIDIA machines. On Mac, install a CPU (or MPS-capable) torch first, then the rest of the stack without re-pulling the CUDA build:

```
python -m pip install torch torchvision
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If the second command tries to replace torch with a `+cu126` wheel and fails, install packages individually from `requirements.txt` skipping the torch lines, or use:

```
python -m pip install torch torchvision
python -m pip install fastapi "uvicorn[standard]" python-multipart anthropic==0.116.0 grad-cam albumentations opencv-python-headless pillow numpy pandas scikit-learn scikit-image matplotlib rich tqdm pydantic
```

Stay in the activated venv for the rest of this guide.

Git LFS is required for the shipped weights. If `weights/bcc_before.pth` is a tiny text pointer instead of a large binary:

```
brew install git-lfs
git lfs install
git lfs pull
```

## 4. API key (for the live trust check)

```
cp .env.example .env
```

Open `.env` and set:

```
ANTHROPIC_API_KEY=your_key_here
```

The key stays on the server. Without a key, scores and heatmaps still work; the trust card fails closed to DEFER.

## 5. Start the app

```
bash run.sh
```

Open http://127.0.0.1:8000

- **Examples** - curated tiles (gallery ships in the repo; full Heidelberg data not required).
- **Compare** - before/after money shot on one positive tile.
- Sidebar model toggle: `before` (corner heat) vs `after` (tissue heat).

Stop the server with Ctrl+C.

## Optional: train / evaluate / results

Full training is developed on NVIDIA CUDA (Windows/Linux). Mac can still re-print committed metrics:

```
python -m scripts.show_results --run-dir experiments/before
python -m scripts.show_results --run-dir experiments/after
```

Training and full evaluation need the Heidelberg tiles (see README **Data**) and a CUDA GPU. Prefer the Windows path or the live Space for the detector train loop.
