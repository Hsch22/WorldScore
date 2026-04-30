import argparse
import importlib
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worldscore.benchmark.helpers import GetHelpers
from worldscore.benchmark.utils.utils import check_model, get_model2type, type2model


def instantiate(config_path: Path):
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    target = config.pop("_target_")
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**config)


def generate_video(generator, model_helper, instance, generation_type):
    if generation_type != "i2v":
        raise ValueError("generate_memworld_worldscore.py only supports i2v.")

    image_path, prompt_list = model_helper.adapt(instance)
    all_generated_frames = []
    for i, prompt in enumerate(prompt_list):
        generated_frames = generator.generate_video(prompt=prompt, image_path=image_path)
        if generation_type == "i2v":
            image_path = model_helper.save_image(generated_frames[-1], image_path, i + 1)
        if i == 0:
            all_generated_frames += generated_frames
        else:
            all_generated_frames += generated_frames[1:]

    model_helper.save(all_generated_frames)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate WorldScore outputs with MemWorld.")
    parser.add_argument("--model_name", default="memworld_wan22_i2v")
    parser.add_argument("--prompt_set", default="")
    parser.add_argument(
        "--visual_movement",
        default="all",
        choices=["all", "static", "dynamic"],
        help="Which WorldScore split to generate.",
    )
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    if not check_model(args.model_name):
        raise ValueError(f"Model not registered: {args.model_name}")

    model_type = get_model2type(type2model)[args.model_name]
    if model_type == "threedgen":
        visual_movements = ["static"]
    elif args.visual_movement == "all":
        visual_movements = ["static", "dynamic"]
    else:
        visual_movements = [args.visual_movement]

    config_path = PROJECT_ROOT / "world_generators" / "configs" / f"{args.model_name}.yaml"
    generator = instantiate(config_path)

    for visual_movement in visual_movements:
        data_instances, helper = GetHelpers(args.model_name, visual_movement, args.prompt_set)
        if args.num_shards > 1:
            data_instances = [
                instance
                for index, instance in enumerate(data_instances)
                if index % args.num_shards == args.shard_id
            ]
        if args.max_instances is not None:
            data_instances = data_instances[: args.max_instances]
        print(
            f"Generating {visual_movement}: {len(data_instances)} instance(s), "
            f"prompt_set={args.prompt_set or '<default>'}, "
            f"shard={args.shard_id}/{args.num_shards}"
        )
        for index, instance in enumerate(data_instances):
            print(f"[{visual_movement}] {index + 1}/{len(data_instances)} -> {instance['output_dir']}")
            generate_video(generator, helper, instance, generation_type="i2v")


if __name__ == "__main__":
    main()
