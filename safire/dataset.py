import logging
from datasets import load_dataset as load_hf_dataset
from safire.utils import pil_image_to_base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def construct_conversation(sample):
    """Top-level function for parallel processing - must be picklable."""
    try:
        logger.debug(f"Processing sample with ID: {sample['id']}")
        # Convert the PIL image to base64 string
        base64_image = pil_image_to_base64(sample['image'])

        user_prompt = f"Answer with the option letter (A, B, C, or D) only from the given choices directly.\n\n"
        user_prompt += f"Question: {sample['question']}\n"
        for option in sample['options']:
            user_prompt += f"{option}\n"
        user_prompt += f"\nAnswer:"
        
        # Create conversation
        conversation_with_img = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": base64_image}},
                {"type": "text", "text": user_prompt}
            ]}
        ]   
        sample['conversation'] = conversation_with_img
     
    except Exception as e:
        logger.error(f"Error processing sample with ID {sample.get('id', 'unknown')}: {e}")
        sample['conversation'] = None

    return sample

def load_dataset(args):
    dataset = args.pop("dataset")
    dataset_subset = args.pop("dataset_subset")
    split = args.pop("split")

    # Load Dataset
    dataset = load_hf_dataset(dataset, dataset_subset, split=split)

    # Add id to each sample
    dataset = dataset.map(lambda x, idx: {"id": idx}, with_indices=True, desc="Adding IDs")

    # Add conversation to each sample
    dataset = dataset.map(construct_conversation, num_proc=8, desc="Constructing conversation")
    return dataset
        