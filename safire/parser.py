from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm import EngineArgs, LLM

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_parser(model="Qwen/Qwen2.5-VL-7B-Instruct", max_model_len=2048):
    parser = FlexibleArgumentParser()

    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model=model, max_model_len=max_model_len)
    
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=512)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    # Add dataset args
    dataset_group = parser.add_argument_group("Dataset parameters")
    dataset_group.add_argument("--dataset", type=str, default='RISys-Lab/SAFIRE_MCVQA', help='Name of the dataset (default: RISys-Lab/SAFIRE_MCVQA)')
    dataset_group.add_argument('--dataset-subset', type=str, default='mcqa', help='Dataset subset (default: mcqa)')
    dataset_group.add_argument('--split', type=str, default='test', help='Dataset split (default: test)')
    dataset_group.add_argument('--output-dir', type=str, default='./outputs', help='Output directory path (default: ./outputs)')
    dataset_group.add_argument('--batch-size', type=int, default=128, help='Batch size for inference (default: 128)')
    dataset_group.add_argument('--resume', action='store_true', help='Resume from latest model output in output-dir if available')

    return parser

def extract_sampling_params(args: dict):
    # Pop arguments not used by LLM
    return {
        "max_tokens": args.pop("max_tokens"),
        "temperature": args.pop("temperature"),
        "top_p": args.pop("top_p"),
        "top_k": args.pop("top_k"),
        # "max_model_len": args.pop("max_model_len")
    }

def create_sampling_params(args: dict, llm: LLM):
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    # max_model_len = args.pop("max_model_len")   
    
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

    return sampling_params