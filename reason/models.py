from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def check_answer(answer: str, correct_answer: str):
    answer = answer.lower().replace(".", "").replace(" ", "")
    correct_answer = correct_answer.lower().replace(".", "").replace(" ", "")
    return answer == correct_answer


class CosmosReason1:
    def __init__(self, tensor_parallel_size=8):
        model_path = "nvidia/Cosmos-Reason1-7B"
        self.processor = AutoProcessor.from_pretrained(model_path)
        # Use tensor parallelism for efficient inference
        kwargs = {
            "limit_mm_per_prompt": {"image": 10, "video": 10},
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": True,  # Disable CUDA graph for better stability
        }
        self.llm = LLM(model=model_path, **kwargs)
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            max_tokens=16384,
        )

    def inference(self, prompt: str, video_path: str):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer in the format: "
                "<think>reasoning</think>\n<answer>answer</answer>.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path, "fps": 4},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        answer = outputs[0].outputs[0].text
        try:
            answer = answer.split("<answer>")[1].split("</answer>")[0].strip()
        except Exception:
            answer = answer + " N/A"
        print(answer)
        return answer
