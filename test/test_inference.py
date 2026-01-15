from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from PIL import Image
import sys
sys.path.append(".")

from safire.utils import pil_image_to_base64

def create_parser(model="Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=2048):
    parser = FlexibleArgumentParser()

    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model=model, max_model_len=max_model_len)
    
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=512)
    sampling_group.add_argument("--temperature", type=float, default=0.01)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    return parser

def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    test_img = "assets/sample.jpeg"

    # Create an LLM
    print("Loading model...")
    print("=" * 80)
    print(f"Model path: {args['model']}")
    print("=" * 80)
    llm = LLM(**args)

    # Create sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    print("Simple Chat without image...")
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"},
    ]
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print(f"Prompt: {conversation}")
    print_outputs(outputs)

    print("=" * 80)
    print("Chat with image...")
    user_prompt = "Please describe this image."
    base64_image = pil_image_to_base64(Image.open(test_img))
    conversation_with_img = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": base64_image}},
            {"type": "text", "text": user_prompt}
        ]}
    ]   
    outputs = llm.chat(conversation_with_img, sampling_params, use_tqdm=False)
    print(f"Prompt: {user_prompt}")
    print_outputs(outputs)

    # Test Batch images
    conversation_with_img_batch = [conversation_with_img for _ in range(5)]
    outputs = llm.chat(conversation_with_img_batch, sampling_params, use_tqdm=True)
    print_outputs(outputs)

if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)