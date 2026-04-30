# WorldScore 环境、模型权重与路径配置梳理

生成时间：2026-04-28  
仓库路径：`/share/project/husicheng/WorldScore`

本文件基于仓库代码、YAML、README、环境准备脚本和当前本地文件状态整理。密钥类配置只记录变量名和用途，不记录明文值。

## 1. 根路径与环境变量

### 1.1 当前 `.env`

当前仓库根目录存在 `.env`，内容如下：

| 变量 | 当前值 | 代码用途 |
| --- | --- | --- |
| `WORLDSCORE_PATH` | `/share/project/husicheng` | `worldscore/benchmark/helpers/__init__.py` 会拼成 `${WORLDSCORE_PATH}/WorldScore/config/...`；`config/base_config.yaml` 也会拼 `${WORLDSCORE_PATH}/WorldScore`。 |
| `MODEL_PATH` | `/share/project/husicheng/WorldScore/models` | `config/model_configs/*.yaml` 的 `runs_root` 根目录；生成和评测输出默认写到 `$MODEL_PATH/<model>/worldscore_output`。 |
| `DATA_PATH` | `/share/project/husicheng/WorldScore/data` | 数据集根目录；代码期望数据集位于 `$DATA_PATH/WorldScore-Dataset`。 |

注意：这里的 `WORLDSCORE_PATH` 是仓库父目录，不是仓库本身；代码会再追加 `/WorldScore`。如果改成仓库根目录，会得到重复路径。

每个新 shell 需要导出：

```bash
export $(grep -v '^#' .env | xargs)
```

### 1.2 密钥与 API 环境变量

| 变量 | 用途 | 来源文件/说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | `worldscore/benchmark/helpers/prompt_generator.py` 使用 OpenAI `gpt-4o` 生成/扩展场景提示。 | 代码会 `load_dotenv(".secrets")`。 |
| `RUNWAYML_API_SECRET` | Gen-3 / RunwayML API。 | `world_generators/gen3.py` 用 `RunwayML()` SDK，README 要求写入 `.secrets`。 |
| `MINIMAX_API_KEY` | Minimax I2V API 鉴权。 | `world_generators/minimax.py` 直接读取 `os.environ["MINIMAX_API_KEY"]`。 |
| Hugging Face token | 下载 gated 模型，尤其是 `Vchitect/Vchitect-2.0-2B`。 | README 要求 `huggingface-cli login` 并接受模型协议。 |

仓库当前没有 `.secrets` 文件。README 有一处写成 `.secret`，但顶层 README 和导出命令都使用 `.secrets`。

### 1.3 构建、下载、分布式相关变量

| 变量 | 默认/用途 | 来源 |
| --- | --- | --- |
| `CUDA_HOME` | 评测 README 写 CUDA 12.1；`logs/env_setup/*.sh` 默认 `/usr/local/cuda-12.8`。 | README、环境脚本 |
| `PATH` | 环境脚本会把 `$CUDA_HOME/bin` 放到前面。 | `logs/env_setup/continue_full_eval_env.sh` |
| `MAX_JOBS` | 编译 CUDA 扩展时的并行度，默认 `8`。 | `logs/env_setup/continue_full_eval_env.sh` |
| `TORCH_CUDA_ARCH_LIST` | 默认 `9.0`。 | `logs/env_setup/continue_full_eval_env.sh`、`run_full_eval_validation.sh` |
| `UV_CACHE_DIR` | 当前设置为仓库下 `.uv-cache`。 | 环境脚本 |
| `UV_HTTP_TIMEOUT` | 默认 `180`。 | `continue_full_eval_env.sh` |
| `PYTHONUNBUFFERED` | 评测验证脚本设为 `1`。 | `run_full_eval_validation.sh` |
| `PYTHONPATH` | 验证脚本追加 repo、metrics third_party、SEA-RAFT。 | `run_full_eval_validation.sh` |
| `AM_I_DOCKER=False` | GroundingDINO 安装变量。 | `continue_full_eval_env.sh` |
| `BUILD_WITH_CUDA=True` | GroundingDINO 编译 CUDA 扩展。 | `continue_full_eval_env.sh` |
| `MAMBA_FORCE_BUILD=TRUE` | 安装 `mamba-ssm` 时强制构建。 | `continue_full_eval_env.sh` |
| `RANK` / `WORLD_SIZE` / `LOCAL_RANK` | Wan2.1 分布式推理。默认 `0/1/0`。 | `world_generators/wan.py` |
| `SLURM_*` | 生成/评测可通过 submitit 提交 Slurm。 | `world_generators/generate_videos.py`、`worldscore/benchmark/helpers/evaluator.py` |
| `MS_PROXY` | ModelScope 数据下载代理，默认 `http://10.8.36.21:2080`。 | `logs/env_setup/download_worldscore_dataset.sh` |
| `MS_MAX_WORKERS` | ModelScope 下载并发，默认 `8`。 | `download_worldscore_dataset.sh` |

## 2. 当前本地 Python 环境快照

当前 `.venv/pyvenv.cfg`：

| 项 | 值 |
| --- | --- |
| Python | `3.10.19` |
| venv home | `/share/project/tanhuajie/miniconda3/envs/husicheng/bin` |
| `include-system-site-packages` | `true` |
| `uv` | `0.9.11` |

当前关键包：

| 包 | 版本/位置 |
| --- | --- |
| `worldscore` | `1.0.0`，editable 指向 `/share/project/husicheng/WorldScore` |
| `torch` | `2.9.1`，来自外层 conda env 的 system site-packages |
| `torchvision` | `0.24.1+cu128`，安装在本仓库 `.venv` |
| `torchaudio` | 当前未安装 |

`setup.py` 中 WorldScore 基础依赖：

```text
av, python-dotenv, omegaconf, submitit, structlog, pyfiglet, numpy,
pillow, opencv-python, scipy, fire, hydra-core, requests
```

控制台入口：

```text
worldscore          -> worldscore.cli:main
worldscore-eval     -> worldscore.run_evaluate:main
worldscore-analysis -> worldscore.run_analysis:main
worldscore-gen      -> worldscore.run_generate:main
```

## 3. per-model 依赖环境

| 模型/任务 | 推荐环境/安装文件 | 关键版本/说明 |
| --- | --- | --- |
| 评测主环境 | README + `logs/env_setup/continue_full_eval_env.sh` | README 写 Python 3.10、CUDA 12.1；本地脚本按 CUDA 12.8、torch/torchvision cu128 修补。 |
| DROID-SLAM | README、`thirdparty/DROID-SLAM/environment*.yaml` | 上游 env 是 Python/conda 风格，`pytorch=1.10`、CUDA 11.3；本仓库实际脚本在当前 `.venv` 中编译/安装其扩展。 |
| GroundingDINO | `thirdparty/GroundingDINO/requirements.txt`、环境脚本 | 需要 `BUILD_WITH_CUDA=True` 编译。 |
| SAM2 | `thirdparty/sam2/setup.py` | 环境脚本 `pip install -e thirdparty/sam2 --no-deps`。 |
| VFIMamba | README、环境脚本 | `causal_conv1d`、`mamba_ssm`；当前脚本装 `mamba-ssm==2.3.1`。 |
| CogVideoX | `requirements/cogvideo.txt` | `torch>=2.5.0`、`diffusers>=0.31.0`、`transformers>=4.46.1`。 |
| VideoCrafter / DynamiCrafter | `requirements/crafter.txt` | `pytorch_lightning==1.8.3`、`transformers==4.25.1` 等。 |
| T2V-Turbo | `requirements/t2vturbo.txt` | README 额外安装 `torch==2.5.0` cu121、`flash-attn`、flash-attention 子模块扩展。 |
| Vchitect-2.0 | `requirements/vchitect.txt` | README 额外安装 `torch==2.5.0` cu121。 |
| EasyAnimate | `requirements/easyanimate.txt` | `diffusers>=0.30.1`、`accelerate>=0.25.0`。 |
| Allegro-TI2V | `requirements/allegro.txt` | README 额外安装 `torch==2.4.1`、`torchvision==0.19.1` cu124。 |
| Gen-3 | `requirements/gen_3.txt` | `runwayml`, `requests`。 |
| Minimax | `requirements/minimax.txt` | `requests`。 |
| Wan2.1 | `requirements/wan.txt` | `torch>=2.4.0`、`transformers>=4.49.0`、`dashscope`、`flash_attn`。 |

## 4. thirdparty 子模块

`.gitmodules` 注册的子模块：

```text
thirdparty/DROID-SLAM                  https://github.com/princeton-vl/DROID-SLAM.git
thirdparty/GroundingDINO               https://github.com/IDEA-Research/GroundingDINO.git
thirdparty/sam2                        https://github.com/haoyi-duan/sam2.git
thirdparty/VideoCrafter                https://github.com/AILab-CVC/VideoCrafter.git
thirdparty/DynamiCrafter               https://github.com/Doubiiu/DynamiCrafter.git
thirdparty/flash-attention             https://github.com/Dao-AILab/flash-attention.git
thirdparty/t2v_turbo                   https://github.com/Ji4chenLi/t2v-turbo.git
thirdparty/Vchitect2                   https://github.com/Vchitect/Vchitect-2.0.git
thirdparty/Allegro                     https://github.com/rhymes-ai/Allegro.git
thirdparty/EasyAnimate                 https://github.com/aigc-apps/EasyAnimate.git
thirdparty/Grounded-Segment-Anything   https://github.com/IDEA-Research/Grounded-Segment-Anything.git
thirdparty/SEA-RAFT                    https://github.com/princeton-vl/SEA-RAFT.git
thirdparty/LTX-Video                   https://github.com/Lightricks/LTX-Video.git
thirdparty/Wan2.1                      https://github.com/Wan-Video/Wan2.1.git
```

当前本地为空的子模块目录：

```text
thirdparty/Allegro
thirdparty/DynamiCrafter
thirdparty/EasyAnimate
thirdparty/LTX-Video
thirdparty/SEA-RAFT
thirdparty/Vchitect2
thirdparty/VideoCrafter
thirdparty/Wan2.1
thirdparty/flash-attention
thirdparty/t2v_turbo
```

评测路径下已有一份 vendored metrics 代码，例如 `worldscore/benchmark/metrics/third_party/SEA-RAFT`、`sam2`、`groundingdino`，不完全依赖 `thirdparty/*` 目录。

## 5. 数据集配置

### 5.1 代码期望路径

`config/base_config.yaml`：

```yaml
focal_length: 500
benchmark_root: ${oc.env:WORLDSCORE_PATH}/WorldScore
dataset_root: ${oc.env:DATA_PATH}/WorldScore-Dataset
output_dir: worldscore_output
regenerate: False
frames: 50
```

当前解析后：

```text
benchmark_root = /share/project/husicheng/WorldScore
dataset_root   = /share/project/husicheng/WorldScore/data/WorldScore-Dataset
output_dir     = worldscore_output
```

### 5.2 下载来源

| 脚本 | 来源 | 输出 |
| --- | --- | --- |
| `download.py` | Hugging Face `Howieeeee/WorldScore` 的 `WorldScore-Dataset.zip` | 解压到 `$DATA_PATH` |
| `logs/env_setup/download_worldscore_dataset.sh` | ModelScope dataset `Jasonhsc/WorldArena_WorldScore`，include `*.zip` | `$DATA_PATH/WorldScore-Dataset.zip`，解压到 `$DATA_PATH` |

当前数据集顶层文件：

```text
WorldScore-Dataset/static/static.json
WorldScore-Dataset/static/photorealistic.json
WorldScore-Dataset/static/stylized.json
WorldScore-Dataset/dynamic/dynamic.json
WorldScore-Dataset/dynamic/photorealistic.json
WorldScore-Dataset/dynamic/stylized.json
```

## 6. 模型注册与评测输出配置

模型类别来自 `worldscore/benchmark/utils/modeltype.py`：

| 类别 | 模型 |
| --- | --- |
| `threedgen` | `wonderjourney`, `wonderworld`, `scenescape`, `text2room`, `luciddreamer`, `invisible_stitch` |
| `fourdgen` | `4dfy` |
| `videogen` | `cogvideox_2b_t2v`, `cogvideox_5b_i2v`, `cogvideox_5b_t2v`, `videocrafter1_t2v`, `videocrafter1_i2v`, `videocrafter2_t2v`, `dynamicrafter_512_i2v`, `dynamicrafter_1024_i2v`, `t2v_turbo_t2v`, `vchitect_2_t2v`, `easyanimate_i2v`, `allegro_ti2v`, `gen_3_i2v`, `minimax_i2v`, `ltx_video_i2v`, `wan2.1_i2v` |

`config/model_configs/*.yaml` 负责评测输出路径、分辨率、帧数和 fps：

| 模型 | 类型 | `runs_root` | 分辨率 | 生成类型 | 帧数/FPS | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| `4dfy` | 4D | `$MODEL_PATH/4dfy` | `256x256` | `t2v` | `120 / 30` |  |
| `allegro_ti2v` | Video | `$MODEL_PATH/Allegro` | `1280x720` | `i2v` | `88 / 15` |  |
| `cogvideox_2b_t2v` | Video | `$MODEL_PATH/CogVideoX_2b_t2v` | `720x480` | `t2v` | `49 / 8` |  |
| `cogvideox_5b_i2v` | Video | `$MODEL_PATH/CogVideoX_5b_i2v` | `720x480` | `i2v` | `49 / 8` |  |
| `cogvideox_5b_t2v` | Video | `$MODEL_PATH/CogVideoX_5b_t2v` | `720x480` | `t2v` | `49 / 8` |  |
| `dynamicrafter_1024_i2v` | Video | `$MODEL_PATH/DynamiCrafter_1024_i2v` | `1024x576` | `i2v` | `50 / 10` |  |
| `dynamicrafter_512_i2v` | Video | `$MODEL_PATH/DynamiCrafter_512_i2v` | `512x320` | `i2v` | `50 / 10` |  |
| `easyanimate_i2v` | Video | `$MODEL_PATH/EasyAnimate` | `1344x768` | `i2v` | `49 / 8` |  |
| `gen_3_i2v` | API Video | `$MODEL_PATH/Gen3` | `1280x768` | `i2v` | `253 / 24` |  |
| `invisible_stitch` | 3D | `$MODEL_PATH/invisible_stitch` | `512x512` | `i2v` | 未配置 | `camera_speed=2`, `camera_type=pytorch3d` |
| `ltx_video_i2v` | Video | `$MODEL_PATH/LTX-Video` | `768x512` | `i2v` | `121 / 24` |  |
| `luciddreamer` | 3D | `$MODEL_PATH/LucidDreamer` | `512x512` | `i2v` | 未配置 | `camera_speed=2`, `camera_type=pytorch3d_tensor` |
| `minimax_i2v` | API Video | `$MODEL_PATH/Minimax` | `1072x720` | `i2v` | `141 / 25` |  |
| `scenescape` | 3D | `$MODEL_PATH/SceneScape` | `512x512` | `t2v` | 未配置 | `camera_speed=0.05`, `camera_type=pytorch3d` |
| `t2v_turbo_t2v` | Video | `$MODEL_PATH/t2v_turbo_t2v` | `512x320` | `t2v` | `48 / 16` |  |
| `text2room` | 3D | `$MODEL_PATH/text2room` | `512x512` | `i2v` | 未配置 | `camera_speed=1.0`, `camera_type=tensor` |
| `vchitect_2_t2v` | Video | `$MODEL_PATH/Vchitect-2.0` | `768x432` | `t2v` | `40 / 8` |  |
| `videocrafter1_i2v` | Video | `$MODEL_PATH/VideoCrafter1_i2v` | `512x320` | `i2v` | `16 / 8` |  |
| `videocrafter1_t2v` | Video | `$MODEL_PATH/VideoCrafter1_t2v` | `1024x576` | `t2v` | `16 / 8` |  |
| `videocrafter2_t2v` | Video | `$MODEL_PATH/VideoCrafter2_t2v` | `512x320` | `t2v` | `16 / 8` |  |
| `wan2.1_i2v` | Video | `$MODEL_PATH/Wan2.1` | `832x480` | `i2v` | `81 / 16` |  |
| `wonderjourney` | 3D | `$MODEL_PATH/WonderJourney` | `512x512` | `i2v` | 未配置 | `camera_speed=0.0005`, `camera_type=pytorch3d` |
| `wonderworld` | 3D | `$MODEL_PATH/WonderWorld` | `512x512` | `i2v` | 未配置 | `camera_speed=0.002`, `camera_type=pytorch3d` |

评测输出统一位于：

```text
${runs_root}/worldscore_output/static/...
${runs_root}/worldscore_output/dynamic/...
${runs_root}/worldscore_output/worldscore.json
```

## 7. 生成模型权重与实例化配置

`world_generators/configs/*.yaml` 负责 Hydra 实例化生成器。当前 `world_generators/checkpoints` 目录不存在或为空；`models/` 下目前只有一个已有评测输出文件。

### 7.1 视频生成器配置

| 模型 | `_target_` | 权重/模型路径 | 下载来源/命令 | 关键参数与备注 |
| --- | --- | --- | --- | --- |
| `allegro_ti2v` | `world_generators.allegro_ti2v.Allegro` | `world_generators/checkpoints/allegro_ti2v` | `huggingface-cli download rhymes-ai/Allegro-TI2V --local-dir world_generators/checkpoints/allegro_ti2v` | 代码期望子目录 `vae`, `text_encoder`, `tokenizer`, `transformer`；bf16 transformer/text encoder，VAE float32。 |
| `cogvideox_2b_t2v` | `world_generators.cogvideox.CogVideoX` | `THUDM/CogVideoX-5b` | Diffusers `from_pretrained` 在线/缓存加载 | YAML 名称是 2B，但 `model_path` 写成 5B；代码因 `model_name` 含 `2b` 使用 DDIM scheduler。 |
| `cogvideox_5b_i2v` | `world_generators.cogvideox.CogVideoX` | `THUDM/CogVideoX-5b-i2v` | Diffusers `from_pretrained` | I2V，DPM scheduler，bf16，默认 seed 42。 |
| `cogvideox_5b_t2v` | `world_generators.cogvideox.CogVideoX` | `THUDM/CogVideoX-5b` | Diffusers `from_pretrained` | T2V，DPM scheduler，bf16。 |
| `dynamicrafter_1024_i2v` | `world_generators.dynamicrafter.DynamiCrafter` | `thirdparty/DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt` | README 写 `wget ... Doubiiu/DynamiCrafter_1024 ...` | YAML 还需要 `thirdparty/DynamiCrafter/configs/inference_1024_v1.0.yaml`；当前子模块为空。README 的下载路径与 YAML 期望路径不一致。 |
| `dynamicrafter_512_i2v` | `world_generators.dynamicrafter.DynamiCrafter` | `thirdparty/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt` | README 写 `wget ... Doubiiu/DynamiCrafter_512 ...` | YAML 还需要 `thirdparty/DynamiCrafter/configs/inference_512_v1.0.yaml`；当前子模块为空。README 的下载路径与 YAML 期望路径不一致。 |
| `easyanimate_i2v` | `world_generators.easyanimate.EasyAnimate` | 默认 `world_generators/checkpoints/easyanimate` | `huggingface-cli download alibaba-pai/EasyAnimateV5-12b-zh-InP --local-dir world_generators/checkpoints/easyanimate` | YAML 未写 `model_path`，类默认值生效；还需要 `thirdparty/EasyAnimate/config/easyanimate_video_v5_magvit_multi_text_encoder.yaml`，当前子模块为空。 |
| `gen_3_i2v` | `world_generators.gen3.Gen3` | 无本地权重 | RunwayML API，模型 ID `gen3a_turbo` | 需要 `RUNWAYML_API_SECRET`。代码上传首帧 base64，轮询任务，下载视频并抽帧。 |
| `ltx_video_i2v` | `world_generators.ltx_video.LTXVideo` | 由 `thirdparty/LTX-Video/configs/ltxv-13b-0.9.7-dev.yaml` 指定 | 代码中若 checkpoint 文件不存在，会从 HF `Lightricks/LTX-Video` 下载到字面目录 `MODEL_DIR` | 当前 `thirdparty/LTX-Video` 为空，配置文件缺失；运行前需初始化子模块或补齐 config。 |
| `minimax_i2v` | `world_generators.minimax.Minimax` | 无本地权重 | Minimax API | URL `https://api.minimaxi.chat/v1/video_generation`，model `video-01`，需要 `MINIMAX_API_KEY`。 |
| `t2v_turbo_t2v` | `world_generators.t2v_turbo.T2VTurbo` | `world_generators/checkpoints/videocrafter_t2v_512_v2.ckpt` + `world_generators/checkpoints/t2v_turbo_unet_lora.pt` | VideoCrafter2 ckpt + `jiachenli-ucsb/T2V-Turbo-VC2` LoRA | 还需要 `thirdparty/t2v_turbo/configs/inference_t2v_512_v2.0.yaml`；当前子模块为空。 |
| `vchitect_2_t2v` | `world_generators.vchitect.Vchitect` | `world_generators/checkpoints/vchitect` | `huggingface-cli download Vchitect/Vchitect-2.0-2B --local-dir world_generators/checkpoints/vchitect` | Gated HF 模型，需要登录并接受协议；当前子模块 `thirdparty/Vchitect2` 为空。 |
| `videocrafter1_i2v` | `world_generators.videocrafter.VideoCrafter` | `world_generators/checkpoints/videocrafter_i2v_512_v1.ckpt` | `https://huggingface.co/VideoCrafter/Image2Video-512/resolve/main/model.ckpt` | 需要 `thirdparty/VideoCrafter/configs/inference_i2v_512_v1.0.yaml`；当前子模块为空。 |
| `videocrafter1_t2v` | `world_generators.videocrafter.VideoCrafter` | `world_generators/checkpoints/videocrafter_t2v_1024_v1.ckpt` | `https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt` | 需要 `thirdparty/VideoCrafter/configs/inference_t2v_1024_v1.0.yaml`。 |
| `videocrafter2_t2v` | `world_generators.videocrafter.VideoCrafter` | `world_generators/checkpoints/videocrafter_t2v_512_v2.ckpt` | `https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt` | 需要 `thirdparty/VideoCrafter/configs/inference_t2v_512_v2.0.yaml`。 |
| `wan2.1_i2v` | `world_generators.wan.Wan` | `./models/Wan2.1-I2V-14B-480P` | `huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./models/Wan2.1-I2V-14B-480P` | task `i2v-14B`，size `832*480`，frames `81`，fps `16`；支持 `RANK/WORLD_SIZE/LOCAL_RANK` 分布式。 |

### 7.2 3D/4D 硬编码适配脚本

`world_generators/README.md` 说明 3D/4D 生成模型主要采用“把脚本复制到外部模型仓库”的适配方式。

| 模型 | 脚本/配置 | 外部权重或在线模型 |
| --- | --- | --- |
| `wonderjourney` | `world_generators/run_wj_worldscore.py`，默认读取外部仓库 `./config/base-config.yaml` 和 `./config/example.yaml`。 | `StableDiffusionInpaintPipeline.from_pretrained(config["stable_diffusion_checkpoint"])`；`AutoencoderKL` 同 checkpoint；OneFormer `shi-labs/oneformer_coco_swin_large`；MiDaS `dpt_beit_large_512.pt`；`util.chatGPT4.TextpromptGen` 可能依赖 OpenAI。 |
| `wonderworld` | `world_generators/run_ww_worldscore.py`，默认读取外部仓库 `./config/base-config.yaml` 和 `./config/example.yaml`。 | `StableDiffusionInpaintPipeline.from_pretrained(config["stable_diffusion_checkpoint"])`；OneFormer `shi-labs/oneformer_ade20k_swin_large`；Marigold depth `prs-eth/marigold-v1-0`；Marigold normals `prs-eth/marigold-normals-v0-1`；可选 SyncDiffusion / SD 2.0 inpaint。 |
| `scenescape`, `text2room`, `luciddreamer`, `invisible_stitch`, `4dfy` | 只有 `config/model_configs/*.yaml` 中的 WorldScore 评测/输出配置。 | 本仓库未提供对应 `world_generators/configs/*.yaml`；需按 README 的硬编码适配方式在外部模型仓库集成。 |

## 8. 评测权重配置

评测权重默认目录：

```text
worldscore/benchmark/metrics/checkpoints
```

### 8.1 当前本地已存在的评测权重

| 文件 | 当前大小 bytes | 用途 |
| --- | ---: | --- |
| `groundingdino_swint_ogc.pth` | `693997677` | GroundingDINO，`object_detection` 和 t2v `motion_accuracy` 中的文本定位。 |
| `sam_vit_h_4b8939.pth` | `2564550879` | SAM ViT-H，代码参数中保留；README 下载项。 |
| `sam2.1_hiera_base_plus.pt` | `323606802` | SAM2 video predictor，`motion_accuracy` 使用。 |
| `sam2.1_hiera_large.pt` | `898083611` | README 下载项；当前主评测代码使用 base-plus ckpt。 |
| `Tartan-C-T-TSKH-spring540x960-M.pth` | `78883175` | SEA-RAFT 光流，`flow_metrics.py`、`flow_aepe_metrics.py`、`motion_accuracy_metrics.py` 使用。 |
| `Tartan-C-T-TSKH-spring540x960-M.safetensors` | `78778760` | 同上，`download_extra_eval_checkpoints.sh` 先下载 safetensors 再转 pth。 |
| `VFIMamba.pkl` | `264414373` | `motion_smoothness_metrics.py` 使用 VFIMamba 插帧模型。 |
| `droid.pth` | `16061701` | DROID-SLAM，`camera_error_metrics.py` 和 `reprojection_error_metrics.py` 使用。 |
| `models/raft-things.pth` | `21108000` | RAFT fallback/alternative metrics 使用；README 的 `models.zip` 解压产物。 |

### 8.2 评测权重下载来源

| 权重 | 下载来源 |
| --- | --- |
| `groundingdino_swint_ogc.pth` | `https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth` |
| `sam_vit_h_4b8939.pth` | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` |
| `models.zip` / `models/raft-things.pth` | README: Dropbox `https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip`；额外脚本也可从 HF `ddrfan/RAFT` 下载 `raft-things.pth`。 |
| `sam2.1_hiera_large.pt` | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` |
| `sam2.1_hiera_base_plus.pt` | `https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt` |
| `VFIMamba.pkl` | `https://huggingface.co/MCG-NJU/VFIMamba_ckpts/resolve/main/ckpt/VFIMamba.pkl` |
| `droid.pth` | Google Drive file id `1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh`，脚本用 `gdown --fuzzy` 下载。 |
| `Tartan-C-T-TSKH-spring540x960-M.safetensors` | `https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/resolve/main/model.safetensors`，脚本转成 `.pth`。 |

### 8.3 评测指标到权重映射

| 指标 | 类 | 权重/模型 |
| --- | --- | --- |
| `camera_control.camera_error` | `CameraErrorMetric` | `droid.pth`，DROID-SLAM 参数内置；相机内参默认 `[500, 500, 256, 256]`。 |
| `object_control.object_detection` | `ObjectDetectionMetric` | GroundingDINO config `GroundingDINO_SwinT_OGC.py` + `groundingdino_swint_ogc.pth`；`bert_base_uncased_path=None`，会从 HF/缓存加载 `bert-base-uncased`。 |
| `content_alignment.clip_score` | `CLIPScoreMetric` | TorchMetrics `CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")`。 |
| `3d_consistency.reprojection_error` | `ReprojectionErrorMetric` | `droid.pth`。 |
| `photometric_consistency.optical_flow_aepe` | `OpticalFlowAverageEndPointErrorMetric` | SEA-RAFT config `spring-M.json` + `Tartan-C-T-TSKH-spring540x960-M.pth`。 |
| `style_consistency.gram_matrix` | `GramMatrixMetric` | 无显式外部权重。 |
| `subjective_quality.clip_iqa+` | `CLIPImageQualityAssessmentPlusMetric` | `pyiqa.create_metric("clipiqa+")`，权重由 pyiqa 自动缓存/下载。 |
| `subjective_quality.clip_aesthetic` | `IQACLIPAestheticScoreMetric` | `pyiqa.create_metric("laion_aes")`，权重由 pyiqa 自动缓存/下载。 |
| `motion_accuracy.motion_accuracy` | `MotionAccuracyMetric` | SEA-RAFT `Tartan...pth`；SAM2 config `sam2.1_hiera_b+.yaml` + `sam2.1_hiera_base_plus.pt`；t2v 情况还会加载 GroundingDINO。 |
| `motion_magnitude.optical_flow` | `OpticalFlowMetric` | SEA-RAFT `Tartan...pth`。 |
| `motion_smoothness.motion_smoothness` | `MotionSmoothnessMetric` | `VFIMamba.pkl`，外加 SSIM/LPIPS。 |

### 8.4 其他导入但非主评测 aspect 使用的权重

| 类/文件 | 权重/模型 | 状态 |
| --- | --- | --- |
| `QAlignVideoMetric` | HF `q-future/one-align` | 在 metrics `__init__` 中导入，但当前 `aspect_info` 未使用。 |
| `CLIPMLPAestheticScoreMetric` | `worldscore/benchmark/metrics/checkpoints/sac+logos+ava1-l14-linearMSE.pth` + OpenAI CLIP `ViT-L/14` | 文件当前不存在；该类被导入但当前 evaluator 未实例化。 |
| `CLIPConsistencyMetric` | OpenAI CLIP `ViT-B/32` | 被导入但当前主 evaluator 未使用。 |
| FlowFormer++ configs | `things_288960.pth` 等 | 代码中存在 alternative metrics/configs，但主 metrics import 当前使用 SEA-RAFT 路径。 |

## 9. 评测标准化参数

标准化参数来自 `worldscore/benchmark/utils/utils.py` 的 `aspect_info`：

| Aspect | Metric | 归一化参数 |
| --- | --- | --- |
| `camera_control` | `camera_error` | `empirical_max=[15, 0.5]`, `empirical_min=[0, 0]`, 越低越好。 |
| `object_control` | `object_detection` | `[0, 1]`，越高越好。 |
| `content_alignment` | `clip_score` | `avg=26.67`, `std=0.8875`, `z_min=-1.5950`, `z_max=1.2741`, range `[0.25, 0.75]`。 |
| `3d_consistency` | `reprojection_error` | `empirical_max=1.0719`, `empirical_min=0`, 越低越好。 |
| `photometric_consistency` | `optical_flow_aepe` | `empirical_max=1.1920`, `empirical_min=0`, 越低越好。 |
| `style_consistency` | `gram_matrix` | `empirical_max=0.0070`, `empirical_min=0`, 越低越好。 |
| `subjective_quality` | `clip_iqa+` | `avg=0.5842`, `std=0.0441`, `z_min=-1.8342`, `z_max=1.8703`, range `[0.25, 0.75]`。 |
| `subjective_quality` | `clip_aesthetic` | `avg=5.5952`, `std=0.2561`, `z_min=-2.9342`, `z_max=1.9741`, range `[0.25, 0.75]`。 |
| `motion_accuracy` | `motion_accuracy` | `avg=-0.1965`, `std=2.3687`, `z_min=-1.6147`, `z_max=1.5180`, range `[0.25, 0.75]`。 |
| `motion_magnitude` | `optical_flow` | `avg=3.2425`, `std=3.4505`, `z_min=-0.8638`, `z_max=2.7498`, range `[0.25, 0.75]`。 |
| `motion_smoothness` | `motion_smoothness` | `empirical_max=[82.4014, 1, 0.0228]`, `empirical_min=[0, 0.9224, 0]`；MSE/LPIPS 越低越好，SSIM 越高越好。 |

## 10. 当前本地模型/输出状态

| 路径 | 当前状态 |
| --- | --- |
| `worldscore/benchmark/metrics/checkpoints` | 已有评测权重，见第 8 节。 |
| `world_generators/checkpoints` | 当前未发现文件。 |
| `models/Wan2.1-I2V-14B-480P` | 当前未发现权重文件。 |
| `models/CogVideoX_5b_i2v/worldscore_output/worldscore.json` | 当前存在一个评测结果文件，大小 `377` bytes。 |
| `data/WorldScore-Dataset` | 当前存在 static/dynamic JSON 数据。 |

## 11. 需要注意的配置不一致/缺失点

1. `WORLDSCORE_PATH` 当前必须保持为仓库父目录 `/share/project/husicheng`，因为代码会追加 `/WorldScore`。
2. `world_generators/configs/cogvideox_2b_t2v.yaml` 的 `model_path` 写的是 `THUDM/CogVideoX-5b`，与模型名 `2b` 不一致。
3. DynamiCrafter README 的下载目标是 `world_generators/checkpoints/*.ckpt`，但 YAML 期望 `thirdparty/DynamiCrafter/checkpoints/.../model.ckpt`。
4. `thirdparty/LTX-Video` 当前为空，`world_generators/configs/ltx_video_i2v.yaml` 指向的 pipeline config 文件不存在。
5. 多个生成模型子模块当前为空，运行前需要 `git submodule update --init ...` 或手动放入外部项目。
6. `CLIPMLPAestheticScoreMetric` 需要 `sac+logos+ava1-l14-linearMSE.pth`，当前未发现该文件；不过主 evaluator 当前没有实例化这个 metric。
7. API 生成器和 OpenAI prompt helper 需要 `.secrets`；当前仓库没有 `.secrets`。
