import json
import os
import re
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，"
    "静止不动的画面，杂乱的背景"
)


class MemWorldWan22:
    def __init__(
        self,
        model_name: str,
        generation_type: Literal["i2v"],
        memworld_root: str,
        model_dir: str,
        dit_ckpt_path: str,
        memory_dir: str,
        height: int = 352,
        width: int = 640,
        frames: int = 257,
        fps: int = 30,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        seed: int = 42,
        device: str = "cuda",
        memory_mode: str = "auto",
        memory_max_size: int = 500,
        memory_update_stride: int = 4,
        origin_memory_index: int = 0,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        allow_repeat_memory_context: bool = False,
        tiled: bool = False,
    ):
        if generation_type != "i2v":
            raise ValueError("MemWorldWan22 currently supports i2v generation only.")
        if frames % 32 != 1:
            raise ValueError(f"MemWorld requires frames % 32 == 1, got {frames}.")

        self.model_name = model_name
        self.generation_type = generation_type
        self.memworld_root = Path(memworld_root)
        self.model_dir = Path(model_dir)
        self.dit_ckpt_path = Path(dit_ckpt_path)
        self.memory_dir = Path(memory_dir)
        self.height = height
        self.width = width
        self.frames = frames
        self.fps = fps
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.device = device
        self.memory_mode = memory_mode
        self.memory_max_size = memory_max_size
        self.memory_update_stride = memory_update_stride
        self.origin_memory_index = origin_memory_index
        self.negative_prompt = negative_prompt
        self.allow_repeat_memory_context = allow_repeat_memory_context
        self.tiled = tiled

        self._check_required_paths()
        if str(self.memworld_root) not in sys.path:
            sys.path.insert(0, str(self.memworld_root))

        from utils.model_loading import build_memory_pipeline

        self.pipe = build_memory_pipeline(
            model_dir=self.model_dir,
            device=self.device,
            include_dit=True,
            include_text_encoder=True,
            include_vae=True,
        )
        self.pipe.initialize_memory_modules(enable_keypoints=True)
        print(f"Loading DiT checkpoint: {self.dit_ckpt_path}")
        self.pipe.load_memory_state_dict(str(self.dit_ckpt_path), strict=True)
        self.pipe.to(device=self.device, dtype=torch.bfloat16)
        self.pipe.device = self.device
        self.pipe.torch_dtype = torch.bfloat16

        self.memory_latents, self.memory_c2ws = self._load_and_encode_memory()
        if not 0 <= self.origin_memory_index < len(self.memory_c2ws):
            raise ValueError(
                f"origin_memory_index={self.origin_memory_index} is out of range "
                f"for {len(self.memory_c2ws)} memory frames."
            )

    def _check_required_paths(self):
        required = [
            self.memworld_root,
            self.model_dir,
            self.dit_ckpt_path,
            self.memory_dir / "frames",
            self.memory_dir / "c2ws.npy",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing MemWorld inputs:\n" + "\n".join(missing))

    def _load_and_encode_memory(self):
        frames_dir = self.memory_dir / "frames"
        c2ws_path = self.memory_dir / "c2ws.npy"

        memory_c2ws = np.load(str(c2ws_path))
        frame_files = sorted(frames_dir.glob("*.png"))
        if len(frame_files) != len(memory_c2ws):
            raise ValueError(
                f"Frame count {len(frame_files)} != c2ws count {len(memory_c2ws)}"
            )

        print(f"Encoding {len(frame_files)} memory frames ...")
        memory_latents = []
        self.pipe.load_models_to_device(["vae"])
        for frame_file in tqdm(frame_files, desc="Encoding memory"):
            img = Image.open(frame_file).convert("RGB").resize(
                (self.width, self.height), resample=Image.BICUBIC
            )
            img_tensor = self.pipe.preprocess_image(img).transpose(0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latent = self.pipe.encode_video(img_tensor)[0]
            memory_latents.append(latent.cpu())

        print(f"Memory loaded: {len(memory_latents)} frames, c2ws {memory_c2ws.shape}")
        return memory_latents, memory_c2ws

    def _segment_index(self, image_path: str) -> int:
        stem = Path(image_path).stem
        if stem == "input_image":
            return 0
        match = re.fullmatch(r"input_image_(\d+)", stem)
        if match:
            return int(match.group(1))
        return 0

    def _resolve_memory_mode(self, image_data: dict) -> str:
        if self.memory_mode != "auto":
            return self.memory_mode
        return "dynamic" if image_data.get("visual_movement") == "dynamic" else "static"

    def _align_to_memory_origin(self, c2ws: np.ndarray, reference_c2w: np.ndarray) -> np.ndarray:
        origin_c2w = np.asarray(self.memory_c2ws[self.origin_memory_index], dtype=np.float64)
        reference_c2w = np.asarray(reference_c2w, dtype=np.float64)
        inv_reference = np.linalg.inv(reference_c2w)
        aligned = []
        for c2w in np.asarray(c2ws, dtype=np.float64):
            relative = inv_reference @ c2w
            aligned.append(origin_c2w @ relative)
        return np.stack(aligned, axis=0)

    def _fixed_c2ws(self) -> np.ndarray:
        origin = np.asarray(self.memory_c2ws[self.origin_memory_index], dtype=np.float64)
        return np.repeat(origin[None], self.frames, axis=0)

    def _target_c2ws(self, output_dir: Path, image_data: dict, image_path: str) -> np.ndarray:
        if image_data.get("visual_movement") == "static":
            camera_data_path = output_dir / "camera_data.json"
            if not camera_data_path.exists():
                raise FileNotFoundError(f"{camera_data_path} not found for static sample.")
            with open(camera_data_path, encoding="utf-8") as f:
                camera_data = json.load(f)
            cameras_interp = np.asarray(camera_data["cameras_interp"], dtype=np.float64)
            segment_idx = self._segment_index(image_path)
            start = segment_idx * (self.frames - 1)
            end = start + self.frames
            segment = cameras_interp[start:end]
            if len(segment) != self.frames:
                raise ValueError(
                    f"Static camera segment length mismatch: got {len(segment)}, "
                    f"expected {self.frames}, segment={segment_idx}, path={camera_data_path}"
                )
            return self._align_to_memory_origin(segment, cameras_interp[0])

        # WorldScore dynamic data currently uses fixed camera paths.
        return self._fixed_c2ws()

    def generate_video(
        self,
        prompt: str,
        image_path: Optional[str] = None,
    ):
        if image_path is None:
            raise ValueError("MemWorldWan22 requires image_path for i2v generation.")
        output_dir = Path(image_path).parent
        image_data_path = output_dir / "image_data.json"
        if not image_data_path.exists():
            raise FileNotFoundError(f"{image_data_path} not found.")
        with open(image_data_path, encoding="utf-8") as f:
            image_data = json.load(f)

        input_image = Image.open(image_path).convert("RGB").resize(
            (self.width, self.height), resample=Image.BICUBIC
        )
        target_c2ws = self._target_c2ws(output_dir, image_data, image_path)
        memory_mode = self._resolve_memory_mode(image_data)

        video = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            input_image=input_image,
            c2ws=target_c2ws,
            memory_latents=self.memory_latents,
            memory_c2ws=self.memory_c2ws,
            memory_mode=memory_mode,
            memory_max_size=self.memory_max_size,
            memory_update_stride=self.memory_update_stride,
            allow_repeat_memory_context=self.allow_repeat_memory_context,
            viz_dir=None,
            height=self.height,
            width=self.width,
            cfg_scale=self.cfg_scale,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            tiled=self.tiled,
        )
        if len(video) != self.frames:
            raise ValueError(f"Generated {len(video)} frames, expected {self.frames}.")
        return video
