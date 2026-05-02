# WorldScore Reproduction Notes

This file records the environment, data, weights, patches, and path edits needed
to run the current `Hsch22/WorldScore` branch on another machine.

The GitHub repository intentionally does not contain large datasets, checkpoints,
model weights, generated outputs, secrets, or local virtual environments. A plain
`git clone` is therefore not enough.

## 1. Tested Context

The local setup that produced the current repository state used:

| Item | Value / note |
| --- | --- |
| OS | Linux |
| Python | 3.10.x; local snapshot was Python 3.10.19 |
| CUDA | README baseline is CUDA 12.1; local compatibility notes also mention CUDA 12.8 |
| Main eval PyTorch baseline | `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`, CUDA 12.1 wheels |
| Local observed PyTorch snapshot | `torch==2.9.1`, `torchvision==0.24.1+cu128`, no `torchaudio` |
| GPU arch patch | `patches/droid-slam-cu12.patch` compiles DROID-SLAM only for `sm_90` |
| WorldScore dataset path | `$DATA_PATH/WorldScore-Dataset` |
| Eval checkpoints path | `worldscore/benchmark/metrics/checkpoints` |
| MemWorld external repo | not included in this repo; must be cloned/copied separately |

If the new machine is not H100 / compute capability 9.0, adjust the DROID-SLAM
CUDA arch flags in `patches/droid-slam-cu12.patch` before applying it, or edit
`thirdparty/DROID-SLAM/setup.py` after applying.

## 2. Clone The Repository

```bash
git clone --recursive https://github.com/Hsch22/WorldScore.git
cd WorldScore
git submodule update --init --recursive
git checkout main
```

If the repository was cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

## 3. Apply Local Submodule Compatibility Patches

These patches are stored in the parent repo because we do not have write access
to the upstream submodule repositories.

```bash
git -C thirdparty/DROID-SLAM apply ../../patches/droid-slam-cu12.patch
git -C thirdparty/GroundingDINO apply ../../patches/groundingdino-cu12.patch
```

Sanity check:

```bash
git -C thirdparty/DROID-SLAM diff --check
git -C thirdparty/GroundingDINO diff --check
```

The local DROID-SLAM nested submodule checkouts observed during packaging were:

```text
thirdparty/DROID-SLAM/thirdparty/eigen    2f3c27c23a664bfed66d1d1a976de160fac47ae1
thirdparty/DROID-SLAM/thirdparty/lietorch e7df86554156b36846008d8ddbcc4d8521a16554
```

## 4. Create `.env`

`WORLDSCORE_PATH` must be the parent directory of this repo, not the repo itself,
because the code appends `/WorldScore`.

Example if the repo is `/share/project/USER/WorldScore`:

```bash
cat > .env <<'EOF'
WORLDSCORE_PATH=/share/project/USER
MODEL_PATH=/share/project/USER/WorldScore/models
DATA_PATH=/share/project/USER/WorldScore/data
EOF

export $(grep -v '^#' .env | xargs)
```

Do not commit `.env`. It is ignored.

## 5. Create The Python Environment

Recommended baseline for the WorldScore evaluation environment:

```bash
conda create -n worldscore python=3.10 -y
conda activate worldscore

python -m pip install --upgrade pip setuptools wheel ninja packaging

# CUDA 12.1 baseline from upstream README.
pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install --index-url https://download.pytorch.org/whl/cu121 xformers

pip install -e .
pip install \
  yacs loguru einops timm imageio spacy catalogue pyiqa torchmetrics \
  pytorch_lightning cvxpy open3d tensorboard matplotlib pyyaml tqdm gdown
pip install evo --upgrade --no-binary evo
python -m spacy download en_core_web_sm
```

VFIMamba / Mamba dependencies:

```bash
pip install causal_conv1d==1.5.0.post8 mamba_ssm==2.2.4
```

If `mamba_ssm==2.2.4` fails on the target CUDA/PyTorch combination, the local
notes recorded `mamba-ssm==2.3.1` as another workable version. Building from
source is also viable:

```bash
git clone https://github.com/state-spaces/mamba.git /tmp/mamba
pip install /tmp/mamba
```

## 6. Build / Install Third-Party Eval Components

DROID-SLAM:

```bash
export CUDA_HOME=/path/to/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=8

cd thirdparty/DROID-SLAM
python setup.py install
cd ../..
```

Grounding-SAM / GroundingDINO:

```bash
cd thirdparty/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
cd ../..
```

Top-level GroundingDINO, with the local patch applied in section 3:

```bash
cd thirdparty/GroundingDINO
export BUILD_WITH_CUDA=True
pip install --no-build-isolation -e .
cd ../..
```

SAM2:

```bash
cd thirdparty/sam2
pip install -e . --no-deps
cd ../..
```

## 7. Restore Data, Checkpoints, And Existing Outputs

Large assets were uploaded to ModelScope dataset:

```text
https://www.modelscope.cn/datasets/Jasonhsc/move
```

The uploaded archive files are:

| File | Purpose | Size |
| --- | --- | ---: |
| `worldscore_dataset_20260430_131414.tar.zst` | `data/WorldScore-Dataset` | ~1.2 GB |
| `worldscore_eval_required_checkpoints_20260430_131414.tar.zst` | required eval checkpoints | ~1.2 GB |
| `worldscore_experiment_outputs_20260430_131414.tar.zst` | existing CogVideoX validation output | ~1.8 MB |
| `worldscore_code_20260430_131414.tar.zst` | code snapshot from migration time; optional if using GitHub | ~201 MB |
| `eval_required_checkpoints_20260430_131414.files` | checkpoint file list | tiny |

Expected SHA256 values:

```text
44498755b681be695d16aa90fe07a1b5a892a8573f32d119fe9218724f0247e9  worldscore_dataset_20260430_131414.tar.zst
6a0d2336f7a82ead3b2b40e0db41fac0f60c00b071d47f96667ff635395b818e  worldscore_eval_required_checkpoints_20260430_131414.tar.zst
eb2c38f5cff7cf0aa23dee744a893d4c4c46e6718f728320cd7a391e5459ec16  worldscore_experiment_outputs_20260430_131414.tar.zst
7e06bebd9f9fdc842c7f37753cb68da681e0839a36a5871b78a95185d0a48076  worldscore_code_20260430_131414.tar.zst
```

Download the files from the ModelScope web UI or with ModelScope tooling, then
place them in a local staging directory, for example:

```bash
mkdir -p /tmp/worldscore_restore
# Put downloaded *.tar.zst files in /tmp/worldscore_restore.
```

Restore into the cloned repo:

```bash
cd /share/project/USER/WorldScore

sha256sum /tmp/worldscore_restore/*.tar.zst

tar -I zstd -xf /tmp/worldscore_restore/worldscore_dataset_20260430_131414.tar.zst -C .
tar -I zstd -xf /tmp/worldscore_restore/worldscore_eval_required_checkpoints_20260430_131414.tar.zst -C .

# Optional: restores the existing small validation output under models/CogVideoX_5b_i2v.
tar -I zstd -xf /tmp/worldscore_restore/worldscore_experiment_outputs_20260430_131414.tar.zst -C .
```

After restore, these paths should exist:

```text
data/WorldScore-Dataset/static/static.json
data/WorldScore-Dataset/dynamic/dynamic.json
worldscore/benchmark/metrics/checkpoints/groundingdino_swint_ogc.pth
worldscore/benchmark/metrics/checkpoints/sam2.1_hiera_base_plus.pt
worldscore/benchmark/metrics/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth
worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl
worldscore/benchmark/metrics/checkpoints/droid.pth
```

The checkpoint archive intentionally includes only the main evaluator's required
weights. It does not include optional or duplicate fallback files such as:

```text
sam_vit_h_4b8939.pth
sam2.1_hiera_large.pt
Tartan-C-T-TSKH-spring540x960-M.safetensors
models/raft-things.pth
```

## 8. MemWorld External Assets

The MemWorld adapter in this repo expects a separate MemWorld repository and
MemWorld model assets. They are not in this GitHub repo or the WorldScore
migration checkpoint archive.

The current committed config points to this local layout:

```text
memworld_root=/share/project/tanhuajie/MemWorld
model_dir=/share/project/tanhuajie/MemWorld/models/Wan2.2-TI2V-5B
dit_ckpt_path=/share/project/tanhuajie/MemWorld/playground/multi_nodes_train_data_dl3dv_10k_dynamic_only_ue/tensorboard_logs/wan22_wm/version_0/checkpoints/dit_step25000.ckpt
memory_dir=/share/project/tanhuajie/MemWorld/assets/dl3dv_memory
runs_root=/share/project/tanhuajie/MemWorld/output/worldscore_dit_step25000_dl3dv
```

On a new machine, either recreate that layout or edit these two files:

```text
config/model_configs/memworld_wan22_i2v.yaml
world_generators/configs/memworld_wan22_i2v.yaml
```

Required MemWorld paths:

```text
<MEMWORLD_ROOT>/
  utils/model_loading.py
  models/Wan2.2-TI2V-5B/
  playground/.../dit_step25000.ckpt
  assets/dl3dv_memory/
    c2ws.npy
    frames/
      *.png
```

The adapter imports:

```python
from utils.model_loading import build_memory_pipeline
```

So `<MEMWORLD_ROOT>` must be a valid checkout/copy containing that module.

## 9. Smoke Tests

Run from the repo root after `.env`, data, checkpoints, and MemWorld assets are
in place.

Basic imports:

```bash
python - <<'PY'
import torch
import worldscore
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY
```

Check model registration:

```bash
python - <<'PY'
from worldscore.benchmark.utils.utils import check_model
print("memworld_wan22_i2v registered:", check_model("memworld_wan22_i2v"))
PY
```

Generate one static sample:

```bash
export $(grep -v '^#' .env | xargs)
python world_generators/generate_memworld_worldscore.py \
  --model_name memworld_wan22_i2v \
  --visual_movement static \
  --max_instances 1
```

Evaluate one static sample:

```bash
python worldscore/run_evaluate_sharded.py \
  --model_name memworld_wan22_i2v \
  --visual_movement static \
  --num_shards 1 \
  --shard_id 0 \
  --batch_size 1 \
  --max_instances 1
```

Evaluate by shards:

```bash
# Example: run shard 0 of 8 for static.
python worldscore/run_evaluate_sharded.py \
  --model_name memworld_wan22_i2v \
  --visual_movement static \
  --num_shards 8 \
  --shard_id 0 \
  --batch_size 10

# Example: run shard 0 of 8 for dynamic.
python worldscore/run_evaluate_sharded.py \
  --model_name memworld_wan22_i2v \
  --visual_movement dynamic \
  --num_shards 8 \
  --shard_id 0 \
  --batch_size 10
```

Calculate final score after all shards complete:

```bash
python worldscore/run_evaluate_sharded.py \
  --model_name memworld_wan22_i2v \
  --visual_movement static \
  --calculate_worldscore
```

## 10. Common Failure Points

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `Missing MemWorld inputs` | MemWorld paths still point to old machine | Edit both MemWorld YAML files |
| `ModuleNotFoundError: utils.model_loading` | `memworld_root` is wrong or MemWorld repo missing | Point `memworld_root` to the MemWorld checkout |
| DROID-SLAM build fails on GPU arch | patch uses `sm_90` only | Edit DROID-SLAM `setup.py` arch flags |
| GroundingDINO compile error around `value.type()` | local compatibility patch not applied | Apply `patches/groundingdino-cu12.patch` |
| Checkpoint file missing | migration checkpoint archive not restored | Restore `worldscore_eval_required_checkpoints_*.tar.zst` |
| Dataset JSON missing | dataset archive not restored or `DATA_PATH` wrong | Restore dataset archive and check `.env` |
| `worldscore/benchmark/metrics/checkpoints` appears in `git status` | `.gitignore` missing or old branch | Pull latest `main` |

## 11. What Git Contains vs. What It Does Not

Contained in Git:

```text
WorldScore code
MemWorld wrapper and generation script
sharded evaluator
configuration templates
environment and migration notes
third-party compatibility patches
```

Not contained in Git:

```text
data/WorldScore-Dataset
worldscore/benchmark/metrics/checkpoints
models/
MemWorld external repository
Wan2.2 model weights
DiT checkpoint
memory assets
.env
.secrets
local venv / conda env
```

The missing large assets are expected and must be restored explicitly.
