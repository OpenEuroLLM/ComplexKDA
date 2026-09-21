# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import argparse
import glob
import json

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-files", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=200)
    args = parser.parse_args()

    files = sorted(glob.glob(args.data_files))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source_tokens = 0
    sampled_source_tokens = 0
    sampled_qwen_tokens = 0
    sampled_documents = 0

    for path in files:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            counts = parquet.read_row_group(row_group, columns=["token_count"])["token_count"]
            source_tokens += pc.sum(counts).as_py()

        sample = parquet.read_row_group(0, columns=["text", "token_count"])
        sample_count = min(args.samples_per_shard, sample.num_rows)
        indices = np.linspace(0, sample.num_rows - 1, sample_count, dtype=np.int64)
        texts = [sample["text"][index].as_py() for index in indices]
        counts = [sample["token_count"][index].as_py() for index in indices]
        encoded = tokenizer(texts, add_special_tokens=False, return_length=True)
        sampled_source_tokens += sum(counts)
        sampled_qwen_tokens += sum(encoded["length"])
        sampled_documents += sample_count

    ratio = sampled_qwen_tokens / sampled_source_tokens
    print(
        json.dumps(
            {
                "files": len(files),
                "source_tokens": source_tokens,
                "sampled_documents": sampled_documents,
                "sampled_source_tokens": sampled_source_tokens,
                "sampled_qwen3_tokens": sampled_qwen_tokens,
                "qwen3_to_source_ratio": ratio,
                "estimated_qwen3_tokens": round(source_tokens * ratio),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
