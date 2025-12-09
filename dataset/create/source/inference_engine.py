from ..dispatch_engine import AsyncDPDispathEngine


class AsyncDPDispathEngineForThink(AsyncDPDispathEngine):
    def _make_task(self, sample):
        thinking_prompt = (
            "Think and solve the following question step by step. "
            "Please put your thinking and analysis procedure within <think></think>. "
            "Put ONLY your final answer within <answer></answer>.\n"
        )
        normal_prompt = "Put ONLY your final answer within <answer></answer>."

        if "prompt_type" in sample and sample["prompt_type"] == "normal":
            prompt = normal_prompt
        else:
            prompt = thinking_prompt

        sample.update(
            {
                "prompt": [
                    *[
                        {
                            "type": "image",
                            "image": image_pil,
                        }
                        for image_pil in sample["images"]
                    ],
                    {
                        "type": "text",
                        "text": prompt + sample["problem"],
                    },
                ]
            }
        )

        return sample