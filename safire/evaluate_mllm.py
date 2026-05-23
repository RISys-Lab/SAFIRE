import logging
import os
import json
import time
import datetime
from collections import defaultdict
from tqdm import tqdm

from vllm import LLM, EngineArgs

from safire.parser import create_parser, extract_sampling_params, create_sampling_params
from safire.utils import pil_image_to_base64, base64_to_pil_image, image_to_base64_str, extract_answer_letter, parse_model_output
from safire.dataset import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _parse_timestamp_from_name(name: str):
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        return None
    date_part, time_part = parts[1], parts[2]
    if len(date_part) != 8 or len(time_part) != 6:
        return None
    if not (date_part.isdigit() and time_part.isdigit()):
        return None
    return f"{date_part}_{time_part}"


def _find_latest_output_file(output_dir: str, output_model_name: str):
    if not os.path.isdir(output_dir):
        return None
    prefix = f"{output_model_name}_"
    candidates = []
    for name in os.listdir(output_dir):
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            candidates.append(path)
    if not candidates:
        return None

    def sort_key(path: str):
        base = os.path.splitext(os.path.basename(path))[0]
        ts = _parse_timestamp_from_name(base)
        if ts:
            try:
                ts_dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M%S")
                return (1, ts_dt, os.path.getmtime(path))
            except ValueError:
                pass
        return (0, datetime.datetime.min, os.path.getmtime(path))

    return max(candidates, key=sort_key)


def _update_accuracy_stats(output_dict, total_correct, total_count, scenario_stats):
    pred_text = None
    outputs = output_dict.get("outputs") or []
    if outputs:
        first_output = outputs[0]
        if isinstance(first_output, dict):
            pred_text = first_output.get("text")
        else:
            pred_text = getattr(first_output, "text", None)
    ans_text = output_dict.get("answer")

    parsed_pred = parse_model_output(pred_text)
    parsed_ans = extract_answer_letter(ans_text)
    is_correct = parsed_pred == parsed_ans

    scenario = output_dict.get("scenario", "unknown")
    scenario_stats[scenario]["count"] += 1
    if is_correct:
        total_correct += 1
        scenario_stats[scenario]["correct"] += 1
    total_count += 1
    return total_correct, total_count


def _load_resume_state(output_file_path: str):
    completed_by_id = {}
    if not os.path.isfile(output_file_path):
        return set(), 0, 0, defaultdict(lambda: {"correct": 0, "count": 0})

    with open(output_file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping invalid JSON on line %d in %s",
                    line_num,
                    output_file_path,
                )
                continue
            dataset_id = record.get("dataset_id")
            try:
                dataset_id = int(dataset_id)
            except (TypeError, ValueError):
                continue
            record["dataset_id"] = dataset_id
            completed_by_id[dataset_id] = record

    completed_ids = set(completed_by_id.keys())
    total_correct = 0
    total_count = 0
    scenario_stats = defaultdict(lambda: {"correct": 0, "count": 0})
    for record in completed_by_id.values():
        total_correct, total_count = _update_accuracy_stats(
            record,
            total_correct,
            total_count,
            scenario_stats,
        )

    return completed_ids, total_correct, total_count, scenario_stats


def _maybe_patch_glm_processor_args(args: dict) -> None:
    model = (args.get("model") or "").lower()
    if "glm-4.1v" not in model:
        return
    mm_kwargs = args.get("mm_processor_kwargs")
    if mm_kwargs is None:
        mm_kwargs = {}
    if not isinstance(mm_kwargs, dict):
        logger.warning(
            "Skipping GLM mm_processor_kwargs patch; expected dict, got %s",
            type(mm_kwargs).__name__,
        )
        return
    if "temporal_patch_size" not in mm_kwargs:
        mm_kwargs = dict(mm_kwargs)
        mm_kwargs["temporal_patch_size"] = 1
        args["mm_processor_kwargs"] = mm_kwargs
        logger.info("Set mm_processor_kwargs.temporal_patch_size=1 for GLM-4.1V")


def main(args: dict):
    print("Starting MLLM Evaluation")
    # Extract sampling params
    sampling_params = extract_sampling_params(args)

    # Capture output_dir before it is popped by load_dataset
    output_dir = args.get('output_dir', '.')

    # Load Dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(args)
    logger.info("Dataset loaded successfully")
    
    output_dir = args.pop("output_dir")
    batch_size = args.pop("batch_size")
    resume = args.pop("resume", False)

    # Sample Conversation
    logger.info("Sample Conversation:")
    logger.info(dataset['conversation'][0])

    output_model_name = args['model'].replace('/', '_')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = None
    output_timestamp = None
    output_model_name_timestamp = None
    completed_ids = set()

    # Accuracy tracking
    total_correct = 0
    total_count = 0
    scenario_stats = defaultdict(lambda: {"correct": 0, "count": 0})

    if resume:
        output_file_path = _find_latest_output_file(output_dir, output_model_name)
        if output_file_path:
            output_model_name_timestamp = os.path.splitext(os.path.basename(output_file_path))[0]
            parsed_timestamp = _parse_timestamp_from_name(output_model_name_timestamp)
            output_timestamp = parsed_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            completed_ids, total_correct, total_count, scenario_stats = _load_resume_state(
                output_file_path
            )
            logger.info(
                "Resuming from %s with %d completed samples",
                output_file_path,
                len(completed_ids),
            )
        else:
            logger.info(
                "No existing output found for %s in %s; starting a new run",
                output_model_name,
                output_dir,
            )
    if output_file_path is None:
        output_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_model_name_timestamp = f"{output_model_name}_{output_timestamp}"
        output_file_path = os.path.join(output_dir, f"{output_model_name_timestamp}.jsonl")

    if resume and completed_ids and len(completed_ids) >= len(dataset):
        logger.info("All samples already completed; writing results and exiting")
        results_file_path = os.path.join(output_dir, f"{output_model_name_timestamp}-results.json")
        results = {
            "model": args.get('model', 'unknown'),
            "temperature": args.get('temperature', 'default'),
            "top_p": args.get('top_p', 'default'),
            "top_k": args.get('top_k', 'default'),
            "max_tokens": args.get('max_tokens', 'default'),
            "timestamp": output_timestamp,
            "overall_accuracy": total_correct / total_count if total_count > 0 else 0.0,
            "total_samples": total_count,
            "scenario_accuracy": {},
        }
        for scenario, stats in scenario_stats.items():
            acc = stats["correct"] / stats["count"] if stats["count"] > 0 else 0.0
            results["scenario_accuracy"][scenario] = {
                "accuracy": acc,
                "correct": stats["correct"],
                "count": stats["count"],
            }
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Results saved to %s", results_file_path)
        return

    # Load Model
    logger.info("Loading model...")
    logger.info(args)
    _maybe_patch_glm_processor_args(args)
    llm = LLM(**args)
    logger.info("Model loaded successfully")

    sampling_params = create_sampling_params(sampling_params, llm)

    # Run Inference
    logger.info("Running inference...")
    start_time = time.time()

    logger.info(f"Writing outputs to {output_file_path}")
    
    batches = range(0, len(dataset), batch_size)
    for i in tqdm(batches, total=len(batches), desc="Evaluating"):
        batch_slice = dataset[i : i + batch_size]
        batch = batch_slice['conversation']
        batch_ids = batch_slice['id']

        pending = [
            (conv, dataset_id)
            for conv, dataset_id in zip(batch, batch_ids)
            if dataset_id not in completed_ids
        ]
        if not pending:
            continue

        batch = [conv for conv, _ in pending]
        batch_ids = [dataset_id for _, dataset_id in pending]
        batch_outputs = llm.chat(batch, sampling_params, use_tqdm=False)
        
        with open(output_file_path, "a") as f:
            for params, dataset_id in zip(batch_outputs, batch_ids):
                # Manually serialize RequestOutput
                output_dict = {
                    "dataset_id": dataset_id,
                    "image_name": dataset[dataset_id]['image_name'],
                    "scenario": dataset[dataset_id]['scenario'],
                    "question": dataset[dataset_id]['question'],
                    "options": str(dataset[dataset_id]['options']),
                    "answer": dataset[dataset_id]['answer'],
                    "outputs": [
                        {
                            "text": o.text,
                            "finish_reason": o.finish_reason
                        } for o in params.outputs
                    ],
                    "finished": params.finished
                }
                f.write(json.dumps(output_dict) + "\n")

                total_correct, total_count = _update_accuracy_stats(
                    output_dict,
                    total_correct,
                    total_count,
                    scenario_stats,
                )

    # Calculate final stats
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
    
    results = {
        "model": args.get('model', 'unknown'),
        "temperature": args.get('temperature', 'default'),
        "top_p": args.get('top_p', 'default'),
        "top_k": args.get('top_k', 'default'),
        "max_tokens": args.get('max_tokens', 'default'),
        "timestamp": output_timestamp,
        "overall_accuracy": overall_accuracy,
        "total_samples": total_count,
        "scenario_accuracy": {}
    }
    
    for scenario, stats in scenario_stats.items():
        acc = stats["correct"] / stats["count"] if stats["count"] > 0 else 0.0
        results["scenario_accuracy"][scenario] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "count": stats["count"]
        }

    # Save results
    results_file_path = os.path.join(output_dir, f"{output_model_name_timestamp}-results.json")
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_file_path}")


    end_time = time.time()
    logger.info("Inference completed successfully")
    logger.info(f"Inference time: {end_time - start_time} seconds")
    logger.info("Outputs saved successfully")


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)