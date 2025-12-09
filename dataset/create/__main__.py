import argparse
from pathlib import Path
from time import sleep

import ray

from .dump_engine import DumpEngine
from .sglang_worker import SGLangWorker
from .source.ai2d import AsyncDPDispathEngineForAI2D
from .source.codeforces import AsyncDPDispathEngineForCodeforces
from .source.docvqa import AsyncDPDispathEngineForDocVQA
from .source.infographicvqa import AsyncDPDispathEngineForInfographicVQA
from .source.mathvision import AsyncDPDispathEngineForMathVision
from .source.mmmu import (
    AsyncDPDispathEngineForMMMUBio,
    AsyncDPDispathEngineForMMMUChem,
    AsyncDPDispathEngineForMMMUCS,
    AsyncDPDispathEngineForMMMUGeo,
    AsyncDPDispathEngineForMMMUPhys,
)
from .source.mmstar import AsyncDPDispathEngineForMMStar
from .source.ocrbench import AsyncDPDispathEngineForOCRBench
from .source.pixmo_count import AsyncDPDispathEngineForPixmoCount
from .source.refl4 import AsyncDPDispathEngineForRefL4
from .source.sat2 import AsyncDPDispathEngineForSAT2
from .source.treebench import AsyncDPDispathEngineForTreeBench
from .source.unisvg import AsyncDPDispathEngineForUniSVG
from .source.virgorlsa import AsyncDPDispathEngineForVIRGORLSA
from .source.virl39k import AsyncDPDispathEngineForViRL39K
from .source.webcode import AsyncDPDispathEngineForWebcode

# Data Item:
# {
#     "id":           str,                                       // {source}_{idx}
#     "images":       list[PIL.Image],
#     "problem":      str,                                       // with <image> in the problem
#     "problem_type": enum["stem", "ocr", "general", "grounding", "coding", "embodied"],
#     "answer":       list[str],                                 // list[str] for some OCR dataset
#     "answer_type":  enum["multiple-choice", "math-expression", "html-code",
#     "answer_type":  enum["multiple-choice", "math-expression", "html-code",
#                          "general-code", "bbox", "number", "critic", "bool", "ocrtext"],
#     "source":       str
# }

DEFAULT_THRESHOLDS = {
    "mathvision": (0.6, 0.0),
    "mmmu_bio": (2, -1),
    "mmmu_chem": (2, -1),
    "mmmu_phys": (2, -1),
    "mmmu_geo": (2, -1),
    "mmmu_cs": (2, -1),
    "docvqa": (0.6, 0.0),
    "infographicvqa": (0.6, 0.0),
    "ocrbench": (0.6, 0.0),
    "mmstar": (0.6, 0.0),
    "webcode": (0.6, 0.0),
    "codeforces": (2, -1),
    "treebench": (0.6, 0.0),
    "refl4": (0.6, 0.0),
    "refspatial": (0.6, 0.0),
    "virl39k": (0.6, 0.0),
    "sat2": (2, -1),
    "virgorlsa": (2, -1),
    "ai2d": (0.6, 0.0),
    "pixmo_count": (0.6, 0.0),
    "unisvg": (0.6, 0.0),
}

# Dispatch Engine Factory
DISPATCH_ENGINE_REGISTRY = {
    "mathvision": AsyncDPDispathEngineForMathVision,
    "mmmu_bio": AsyncDPDispathEngineForMMMUBio,
    "mmmu_chem": AsyncDPDispathEngineForMMMUChem,
    "mmmu_phys": AsyncDPDispathEngineForMMMUPhys,
    "mmmu_geo": AsyncDPDispathEngineForMMMUGeo,
    "mmmu_cs": AsyncDPDispathEngineForMMMUCS,
    "docvqa": AsyncDPDispathEngineForDocVQA,
    "infographicvqa": AsyncDPDispathEngineForInfographicVQA,
    "ocrbench": AsyncDPDispathEngineForOCRBench,
    "mmstar": AsyncDPDispathEngineForMMStar,
    "webcode": AsyncDPDispathEngineForWebcode,
    "codeforces": AsyncDPDispathEngineForCodeforces,
    "treebench": AsyncDPDispathEngineForTreeBench,
    "refl4": AsyncDPDispathEngineForRefL4,
    "virl39k": AsyncDPDispathEngineForViRL39K,
    "sat2": AsyncDPDispathEngineForSAT2,
    "virgorlsa": AsyncDPDispathEngineForVIRGORLSA,
    "ai2d": AsyncDPDispathEngineForAI2D,
    "pixmo_count": AsyncDPDispathEngineForPixmoCount,
    "unisvg": AsyncDPDispathEngineForUniSVG,
}


def create_dispatch_engine(
    
    dataset_name, dataset_path=None, dataset_split=None, data_files=None
, **kwargs
):
    """Factory function to create dispatch engine for a given dataset."""
    if dataset_name not in DISPATCH_ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {', '.join(DISPATCH_ENGINE_REGISTRY.keys())}"
        )

    engine_class = DISPATCH_ENGINE_REGISTRY[dataset_name]
    return engine_class.remote(
        dataset_path=dataset_path,
        dataset_split=dataset_split,
        data_files=data_files,
        **kwargs,
    )


def process_dataset(args, dispatch_engine, dump_engine):
    workers = [
        SGLangWorker.remote(
            args.model_name,
            dispatch_engine,
            dump_engine,
            {
                "n": args.rollout_n,
                "max_new_tokens": args.max_tokens,
            },
        )
        for _ in range(args.num_workers)
    ]
    ray.get([worker.initialize.remote() for worker in workers])
    for worker in workers:
        worker.start.remote()

    print("Processing dataset...")
    while True:
        total_n = ray.get(dump_engine.is_done.remote())
        if total_n is not False:
            break
        sleep(1)

    print("Total number of saved samples:", total_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rollout-n", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="./data/")
    parser.add_argument(
        "--dataset-name", type=str, default="mathvision", help="Name of the dataset"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=None, help="Number of samples to process"
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Custom dataset path"
    )
    parser.add_argument(
        "--dataset-split", type=str, default=None, help="Custom dataset split"
    )
    parser.add_argument(
        "--data-files", type=str, default=None, help="Custom dataset data files"
    )
    parser.add_argument(
        "--min-reward", type=float, default=None, help="Minimum reward threshold"
    )
    parser.add_argument(
        "--max-reward", type=float, default=None, help="Maximum reward threshold"
    )
    parser.add_argument(
        "--shard-size", type=int, default=2000, help="Number of samples per shard"
    )
    # Ref spatial specific args for selecting sub-case JSONs
    parser.add_argument(
        "--refspatial-specific",
        default=["all"],
        help="One or more specific sub-case JSONs to parse (e.g., '3D/vacant_qa.json'). Default is 'all'.",
        nargs="+",
    )
    parser.add_argument(
        "--to-parquet-only",
        action="store_true",
        help="Only load data and save to parquet without inference/filtering",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Resume from existing parquet file (skip already processed samples)",
    )

    args = parser.parse_args()

    ray.init(include_dashboard=False)

    dump_engine = DumpEngine.remote(
        min_t=args.min_reward if args.min_reward is not None else DEFAULT_THRESHOLDS[args.dataset_name][1],
        max_t=args.max_reward if args.max_reward is not None else DEFAULT_THRESHOLDS[args.dataset_name][0],
        shard_size=args.shard_size,
        output_dir=Path(args.output_dir) / args.dataset_name,
        n=args.dataset_size if args.dataset_size is not None else float('inf'),
    )
    ray.get(dump_engine.__ray_ready__.remote())

    # Create dispatch engine and wait for initialization
    dispatch_engine = create_dispatch_engine(
        args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
        data_files=args.data_files,
    )


    # Wait for the Ray actor to be fully initialized
    # This ensures the __init__ method has completed before proceeding
    ray.get(dispatch_engine.__ray_ready__.remote())

    process_dataset(args, dispatch_engine, dump_engine)
