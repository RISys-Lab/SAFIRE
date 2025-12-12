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


def main(args: dict):
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

    # Sample Conversation
    logger.info("Sample Conversation:")
    logger.info(dataset['conversation'][0])

    # Load Model
    logger.info("Loading model...")
    logger.info(args)
    llm = LLM(**args)
    logger.info("Model loaded successfully")

    sampling_params = create_sampling_params(sampling_params, llm)

    # Run Inference
    logger.info("Running inference...")
    start_time = time.time()
    
    # Generate output filename
    output_model_name = args['model'].replace('/', '_')
    output_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_name_timestamp = f"{output_model_name}_{output_timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{output_model_name_timestamp}.jsonl")
    

    logger.info(f"Writing outputs to {output_file_path}")
    
    # Accuracy tracking
    total_correct = 0
    total_count = 0
    scenario_stats = defaultdict(lambda: {"correct": 0, "count": 0})
    
    batches = range(0, len(dataset), batch_size)
    for i in tqdm(batches, total=len(batches), desc="Evaluating"):
        batch = dataset[i : i + batch_size]['conversation']
        batch_ids = dataset[i : i + batch_size]['id']
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

                # Calculate Accuracy
                # Calculate Accuracy
                # Assuming the first output is the main one and text is the answer
                pred_text = output_dict["outputs"][0]["text"]
                ans_text = output_dict["answer"]
                
                parsed_pred = parse_model_output(pred_text)
                parsed_ans = extract_answer_letter(ans_text)
                
                # If parsing fails or returns None, it counts as incorrect (None != "A")
                is_correct = parsed_pred == parsed_ans
                
                # Debug logging for potential issues (optional, maybe too verbose)
                # logger.debug(f"Pred: {pred_text} -> {parsed_pred}, Ans: {ans_text} -> {parsed_ans}, Correct: {is_correct}")

                
                if is_correct:
                    total_correct += 1
                    scenario_stats[output_dict["scenario"]]["correct"] += 1
                
                total_count += 1
                scenario_stats[output_dict["scenario"]]["count"] += 1

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