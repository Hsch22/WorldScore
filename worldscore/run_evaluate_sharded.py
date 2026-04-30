import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

from omegaconf import OmegaConf

from worldscore.benchmark.helpers.evaluator import (
    Evaluator,
    check_evaluation_completeness,
    process_batch,
)
from worldscore.benchmark.utils.utils import (
    aspect_info,
    check_model,
    get_model2type,
    type2model,
)
from worldscore.run_evaluate import calculate_worldscore


def load_config(model_name: str, visual_movement: str) -> dict:
    base_config = OmegaConf.load(os.path.join("config", "base_config.yaml"))
    model_config = OmegaConf.load(
        os.path.join("config", "model_configs", f"{model_name}.yaml")
    )
    config = OmegaConf.merge(base_config, model_config)
    config.visual_movement = visual_movement
    return OmegaConf.to_container(config, resolve=True)


def collect_instances(evaluator: Evaluator, aspect_list: list[str], reevaluate: bool):
    instance_attributes = []

    if evaluator.visual_movement == "static":
        visual_styles = sorted(x.name for x in evaluator.root_path.iterdir() if x.is_dir())
        for visual_style in visual_styles:
            visual_style_dir = evaluator.root_path / visual_style
            scene_types = sorted(x.name for x in visual_style_dir.iterdir() if x.is_dir())
            for scene_type in scene_types:
                scene_type_dir = visual_style_dir / scene_type
                categories = sorted(x.name for x in scene_type_dir.iterdir() if x.is_dir())
                for category in categories:
                    category_dir = scene_type_dir / category
                    instances = sorted(x.name for x in category_dir.iterdir() if x.is_dir())
                    for instance in instances:
                        instance_dir = category_dir / instance
                        if not evaluator.data_exists(instance_dir):
                            continue
                        if evaluation_complete(instance_dir, aspect_list) and not reevaluate:
                            continue
                        instance_attributes.append(
                            [visual_style, scene_type, category, instance, instance_dir]
                        )

    elif evaluator.visual_movement == "dynamic":
        visual_styles = sorted(x.name for x in evaluator.root_path.iterdir() if x.is_dir())
        for visual_style in visual_styles:
            visual_style_dir = evaluator.root_path / visual_style
            motion_types = sorted(x.name for x in visual_style_dir.iterdir() if x.is_dir())
            for motion_type in motion_types:
                motion_type_dir = visual_style_dir / motion_type
                instances = sorted(x.name for x in motion_type_dir.iterdir() if x.is_dir())
                for instance in instances:
                    instance_dir = motion_type_dir / instance
                    if not evaluator.data_exists(instance_dir):
                        continue
                    if evaluation_complete(instance_dir, aspect_list) and not reevaluate:
                        continue
                    instance_attributes.append(
                        [visual_style, motion_type, instance, instance_dir]
                    )

    else:
        raise ValueError(f"Unknown visual_movement: {evaluator.visual_movement}")

    return instance_attributes


def evaluation_complete(instance_dir: Path, aspect_list: list[str]) -> bool:
    result_path = instance_dir / "evaluation.json"
    if not result_path.exists():
        return False
    try:
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
    except Exception:
        return False
    if not check_evaluation_completeness(aspect_list, result):
        return False
    for aspect in aspect_list:
        aspect_scores = result.get(aspect, {})
        for metric_name in aspect_info[aspect]["metrics"]:
            metric_scores = aspect_scores.get(metric_name)
            if not metric_scores:
                return False
            if "score" not in metric_scores or "score_normalized" not in metric_scores:
                return False
    return True


def chunked(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_shard(args: Namespace) -> None:
    if not check_model(args.model_name):
        raise ValueError(f"Model not registered: {args.model_name}")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    config = load_config(args.model_name, args.visual_movement)
    evaluator = Evaluator(config)
    aspect_list = evaluator.build_full_aspect_list()
    all_instances = collect_instances(
        evaluator=evaluator,
        aspect_list=aspect_list,
        reevaluate=args.reevaluate,
    )
    shard_instances = [
        instance
        for index, instance in enumerate(all_instances)
        if index % args.num_shards == args.shard_id
    ]
    if args.max_instances is not None:
        shard_instances = shard_instances[: args.max_instances]

    print(
        f"Evaluating {args.visual_movement}: {len(shard_instances)} pending "
        f"instance(s) on shard {args.shard_id}/{args.num_shards}; "
        f"total pending={len(all_instances)}"
    )

    for batch_index, instance_batch in enumerate(
        chunked(shard_instances, args.batch_size), start=1
    ):
        print(
            f"[{args.visual_movement}] shard {args.shard_id}: "
            f"batch {batch_index}, size={len(instance_batch)}"
        )
        process_batch(
            config=config,
            instance_batch=instance_batch,
            aspect_list=aspect_list,
            visual_movement=args.visual_movement,
        )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WorldScore outputs by shard.")
    parser.add_argument("--model_name", default="memworld_wan22_i2v")
    parser.add_argument(
        "--visual_movement",
        choices=["static", "dynamic"],
        required=True,
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument("--calculate_worldscore", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calculate_worldscore:
        model_type = get_model2type(type2model)[args.model_name]
        visual_movements = ["static"] if model_type == "threedgen" else ["static", "dynamic"]
        calculate_worldscore(args.model_name, visual_movements)
        return
    run_shard(args)


if __name__ == "__main__":
    main()
