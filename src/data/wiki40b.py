from transformers import AutoTokenizer
import numpy as np
import torch
from datasets import load_dataset, Dataset
import multiprocessing
from tqdm import tqdm

from utils import get_data_folder

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def download_and_process_wiki40b(subset: str, num_proc: int) -> dict[str, Dataset]:
    def tokenize_function(example: dict) -> dict:
        return tokenizer(example["text"])

    dataset = load_dataset("wiki40b", subset)

    tokenized_dataset: dict[str, Dataset] = {}
    for split in dataset.keys():
        print(f"Tokenizing {split} split")
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            num_proc=multiprocessing.cpu_count() - 1,
            remove_columns=["text", "wikidata_id", "version_id"],
        )

    # Add the eos and bos tokens
    for split in tokenized_dataset.keys():
        print(f"Adding bos and eos tokens to {split} split")
        tokenized_dataset[split] = tokenized_dataset[split].map(
            lambda example: {
                "ids": [tokenizer.bos_token_id]
                + example["input_ids"]
                + [tokenizer.eos_token_id],
                "len": len(example["input_ids"]) + 2,
                "attention_mask": [1, 1] + example["attention_mask"],
            },
            num_proc=multiprocessing.cpu_count() - 1,
        )
    return tokenized_dataset


def get_wiki40b(subset="en", num_proc=40, return_torch=True):
    """https://huggingface.co/datasets/wiki40b"""
    WIKI_40B_PATH = get_data_folder() / "wiki40b"
    SUBSET_PATH = WIKI_40B_PATH / subset
    SUBSET_PATH.mkdir(exist_ok=True, parents=True)

    if not SUBSET_PATH.joinpath("train.bin").exists():
        tokenized_dataset = download_and_process_wiki40b(subset, num_proc=num_proc)

        # now save the tokenized datasets
        for split, dset in tokenized_dataset.items():
            arr_len = np.sum(dset["len"])
            filename = SUBSET_PATH / f"{split}.bin"
            print("saving to ", filename)
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 10

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_path = SUBSET_PATH / "train.bin"
    test_path = SUBSET_PATH / "test.bin"
    val_path = SUBSET_PATH / "validation.bin"

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    test_data = np.memmap(test_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    print(
        f"Subset {subset}: train[{len(train_data)}] | val[{len(val_data)}] | test[{len(test_data)}]"
    )

    num_to_show = 4
    print(f"train: {train_data[:num_to_show]} + [...] + {train_data[-num_to_show:]}")
    print(f"val: {val_data[:num_to_show]} + [...] + {val_data[-num_to_show:]}")
    print(f"test: {test_data[:num_to_show]} + [...] + {test_data[-num_to_show:]}")

    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        test_data = torch.tensor(np.array(test_data, dtype=np.int32))
        val_data = torch.tensor(np.array(val_data, dtype=np.int32))
    return {"train": train_data, "val": val_data, "test": test_data}


get_wiki40b(subset="en", num_proc=multiprocessing.cpu_count() - 1, return_torch=False)
