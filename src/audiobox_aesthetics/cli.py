# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import logging
import itertools
import os
from pathlib import Path
import requests

import submitit
from tqdm import tqdm
from audiobox_aesthetics.infer import load_dataset, main_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_CKPT = "checkpoint.pt"
DEFAULT_CKPT_URL = "https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt"


def parse_args():
    parser = argparse.ArgumentParser("CLI for audiobox-aesthetics inference")
    parser.add_argument("input_file", type=str)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--remote", action="store_true", default=False, help="Set true to run via SLURM"
    )

    # remote == True
    parser.add_argument(
        "--job-dir", default="/tmp", type=str, help="Slurm job directory"
    )
    parser.add_argument(
        "--partition", default="learn", type=str, help="Slurm partition"
    )
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="", type=str, help="Slurm account")
    parser.add_argument("--comment", default="", type=str, help="Slurm job comment")
    parser.add_argument(
        "--constraint",
        default="",
        type=str,
        help="Slurm constraint eg.: ampere80gb For using A100s or volta32gb for using V100s.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Exclude certain nodes from the slurm job.",
    )
    parser.add_argument(
        "--array", default=100, type=int, help="Slurm max array parallelism"
    )
    parser.add_argument(
        "--chunk", default=1000, type=int, help="chunk size per instance"
    )
    return parser.parse_args()


def download_file(url, destination):
    """Download a file from a URL with a progress bar."""
    try:
        # Stream the request to handle large files
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            # Get the total file size from the headers
            total_size = int(response.headers.get("content-length", 0))
            # Open the file in binary write mode
            with open(destination, "wb") as f:
                # Use tqdm to create a progress bar
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(destination),
                ) as pbar:
                    # Iterate over the response in chunks
                    for chunk in response.iter_content(chunk_size=1024):
                        # Write each chunk to the file
                        f.write(chunk)
                        # Update the progress bar
                        pbar.update(len(chunk))
        logging.info(f"File has been downloaded and saved to '{destination}'.")
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while downloading the file: {e}")


def app():
    args = parse_args()

    # Check if checkpoint exists
    if not Path(args.ckpt).exists():
        logging.info(f"{args.ckpt} not found. Downloading  ...")
        download_file(DEFAULT_CKPT_URL, args.ckpt)

    metadata = load_dataset(args.input_file, 0, 2**64)
    fn_wrapped = partial(main_predict, batch_size=args.batch_size, ckpt=args.ckpt)

    if args.remote:
        # chunk metadata
        chunksize = args.chunk
        chunked = [
            metadata[ii : ii + chunksize] for ii in range(0, len(metadata), chunksize)
        ]

        job_dir = Path(args.job_dir)
        job_dir.mkdir(exist_ok=True)

        executor = submitit.AutoExecutor(folder=f"{job_dir}/%A/")

        kwargs = {}
        if len(args.constraint):
            kwargs["slurm_constraint"] = args.constraint
        if args.comment:
            kwargs["slurm_comment"] = args.comment
        if args.qos:
            kwargs["slurm_qos"] = args.qos
        if args.account:
            kwargs["slurm_account"] = args.account

        # Set the parameters for the Slurm job
        executor.update_parameters(
            slurm_nodes=1,
            slurm_gpus_per_node=1,
            slurm_tasks_per_node=1,
            slurm_cpus_per_task=10,
            timeout_min=60 * 20,  # max is 20 hours
            slurm_array_parallelism=min(
                len(chunked), args.array
            ),  # number of tasks in the array job
            slurm_partition=args.partition,
            slurm_exclude=args.exclude,
            **kwargs,
        )

        jobs = executor.map_array(fn_wrapped, chunked)
        outputs = [job.result() for job in jobs]

        outputs = itertools.chain(*outputs)
    else:
        outputs = fn_wrapped(metadata)
    print("\n".join(str(x) for x in outputs))


if __name__ == "__main__":
    """
    Example usage:
    python cli.py input.jsonl --batch-size 100 --ckpt /path/to/ckpt > output.jsonl
    """
    app()
