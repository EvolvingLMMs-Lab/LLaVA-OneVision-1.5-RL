import ray
import sglang as sgl
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


@ray.remote(num_gpus=1)
class SGLangWorker:
    def __init__(self, model_name: str, engine: ray.actor.ActorHandle, dump_engine: ray.actor.ActorHandle, sampling_params: dict):
        self.model_name = model_name
        self.engine = engine
        self.dump_engine = dump_engine
        self.sampling_params = sampling_params
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.still_running = True
        self.submit_ref = []

    def initialize(self):
        self.engine.register.remote(ray.get_runtime_context().get_node_id())
        self.dump_engine.register.remote(ray.get_runtime_context().get_node_id())

        print(f"Worker {ray.get_runtime_context().get_node_id()} initializing LLM {self.model_name}")
        if self.sampling_params["n"] > 1:
            self.llm = sgl.Engine(
                model_path=self.model_name,
                trust_remote_code=True
            )
        else:
            self.llm = None
        print(f"Worker {ray.get_runtime_context().get_node_id()} initialized LLM {self.model_name}")

    def _request(self, req) -> str:
        if self.sampling_params["n"] > 1:
            prompt = req["prompt"]
            conv = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            text = self.processor.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(conv, image_patch_size=self.processor.image_processor.patch_size)

            outputs = self.llm.generate(text, sampling_params=self.sampling_params, image_data=image_inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]
            return [o["text"] for o in outputs]
        else:
            return []

    def start(self):
        print(f"Worker {ray.get_runtime_context().get_node_id()} starting processing loop.")
        while True:
            req = ray.get(self.engine.fetch_task.remote())
            if req is None:
                print("No more tasks received, stopping worker.")
                # no more tasks
                break

            result = self._request(req)
            req["result"] = result
            self.submit_ref.append(self.dump_engine.submit_result.remote(req))

        ray.get(self.submit_ref)
        self.still_running = False
        self.engine.unregister.remote(ray.get_runtime_context().get_node_id())
        self.dump_engine.unregister.remote(ray.get_runtime_context().get_node_id())

    def is_running(self):
        return self.still_running

