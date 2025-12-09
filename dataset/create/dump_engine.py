import os
import time
from pprint import pprint

import ray
from datasets import Dataset, Features, Image, Value

from reward.reward_system import ThinkingRewardSystem

features = Features(
    {
        "id": Value(dtype="string"),
        "images": [Image(decode=True)],
        "problem": Value(dtype="string"),
        "answer": [Value(dtype="string")],
        "problem_type": Value(dtype="string"),
        "answer_type": Value(dtype="string"),
        "source": Value(dtype="string"),
        "prompt_type": Value(dtype="string"),
    }
)

@ray.remote
class DumpEngine:
    def __init__(self, min_t: float, max_t: float, shard_size: int, output_dir: str, n: int):
        self.reward_system = ThinkingRewardSystem(repeat_detection=False)
        self.min_t = min_t
        self.max_t = max_t
        self.shard_size = shard_size
        self.output_dir = output_dir
        self.n = n
        os.makedirs(output_dir, exist_ok=True)

        self.id = 0
        self.submitted_id = 0
        self.buffer = []
        self.finalized = False
        self.start_time = time.time()

        self.workers = []

    def submit_result(self, sample):
        if self.finalized:
            return

        self.submitted_id += 1

        preds = sample["result"]
        if len(preds) > 0:
            rewards = [
                self.reward_system.reward(
                    None, pred, sample["answer"], sample["answer_type"]
                )["reward"]
                for pred in preds
            ]
            reward = sum(rewards) / len(rewards)
        else:
            reward = (self.max_t + self.min_t) / 2

        if self.max_t > reward > self.min_t:
            data_item = {
                "id": f"{sample['source']}_{self.id:04d}",
                "images": [img for img in sample["images"]],
                "problem": sample["problem"],
                "answer": sample["answer"]
                if isinstance(sample["answer"], list)
                else [sample["answer"]],
                "problem_type": sample["problem_type"],
                "answer_type": sample["answer_type"],
                "source": sample["source"],
            }
            if "prompt_type" in sample:
                data_item["prompt_type"] = sample["prompt_type"]
            else:
                data_item["prompt_type"] = "default"

            if self.id % 100 == 0:
                cost_time = time.time() - self.start_time
                eta = max(0, (cost_time / (self.id + 1)) * (self.n - self.id - 1))
                minutes = int(eta // 60)
                seconds = int(eta % 60)
                eta_str = f"{minutes:02d}:{seconds:02d}"
                print(
                    f"[{self.id}/{self.n}] Submitted {self.submitted_id} samples | ETA {eta_str}"
                )

                pprint(data_item)

            self.buffer.append(data_item)

            if len(self.buffer) >= self.shard_size:
                dataset = Dataset.from_list(self.buffer, features=features)
                shard_id = len(list(self.output_dir.glob("*.parquet")))
                dataset.to_parquet(self.output_dir / f"{shard_id:05d}.parquet")
                self.buffer = []

            self.id += 1

            if self.id >= self.n:
                self.finalize()




    def finalize(self):
        self.finalized = True
        self.workers = []
        if len(self.buffer) > 0:
            dataset = Dataset.from_list(self.buffer, features=features)
            shard_id = len(list(self.output_dir.glob("*.parquet")))
            dataset.to_parquet(self.output_dir / f"{shard_id:05d}.parquet")
            self.buffer = []

    def is_done(self):
        if self.finalized:
            return self.id
        return False
        
    def register(self, id: str):
        if id not in self.workers:
            self.workers.append(id)
    
    def unregister(self, id: str):
        if id in self.workers:
            self.workers.remove(id)

        if len(self.workers) == 0:
            self.finalize()