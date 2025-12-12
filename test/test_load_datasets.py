from datasets import load_dataset

def main():
    dataset_name = "RISys-Lab/SAFIRE_MCVQA"
    dataset_subset = "mcqa"
    split= "test"
    print("Loading Dataset")    
    dataset = load_dataset(dataset_name, dataset_subset, split=split, download_mode="force_redownload")
    print(dataset)

if __name__ == "__main__":
    main()