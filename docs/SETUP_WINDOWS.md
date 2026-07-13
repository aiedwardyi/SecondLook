# Setup (Windows)

From a clean machine to the running app. No prior Python setup assumed.

## 1. Install Python 3.12

1. Download the Windows installer from https://www.python.org/downloads/
2. Run it. Check **Add python.exe to PATH**.
3. Open a new PowerShell window and confirm:

```
python --version
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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In cmd.exe use `.\.venv\Scripts\Activate.bat` instead of `Activate.ps1`.

`requirements.txt` pulls CUDA builds of PyTorch when available. A machine without an NVIDIA GPU can still run the app and Grad-CAM on CPU; install may take several minutes either way. Stay in the activated venv for the rest of this guide.

Git LFS is required for the shipped weights. If `weights/bcc_before.pth` is a tiny text pointer instead of a large binary, install Git LFS (https://git-lfs.com) and run:

```
git lfs install
git lfs pull
```

## 4. API key (for the live trust check)

```
copy .env.example .env
```

Open `.env` in a text editor and set:

```
ANTHROPIC_API_KEY=your_key_here
```

The key stays on the server. It is never sent to the browser. Without a key, scores and heatmaps still work; the trust card fails closed to DEFER.

## 5. Start the app

```
.\run.bat
```

Open http://127.0.0.1:8000

- **Examples** - curated tiles (gallery ships in the repo; full Heidelberg data not required).
- **Compare** - before/after attention on one positive tile.
- Sidebar model toggle: `before` (corner heat) vs `after` (tissue heat).

Stop the server with Ctrl+C in the terminal.

## Optional: train / evaluate / results (GPU + full data)

Training and full test-set evaluation need the Heidelberg tiles (see README **Data**) and a CUDA GPU (~10 GB+ VRAM).

```
.\train.bat before
.\train.bat after
.\evaluate.bat before
.\evaluate.bat after
.\results.bat before
.\results.bat after
```

`results.bat` re-prints committed run logs without a GPU. The other three need a completed train (or a local `experiments/<side>/best.pth`).
