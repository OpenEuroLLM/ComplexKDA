# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Stream FineWeb-Edu Parquet shards from Hugging Face into MN5 GPFS.

The download is initiated on the local machine because public MareNostrum 5
nodes cannot make outbound connections. Bytes are piped directly from curl to
an SSH connection, so the local machine only holds bounded pipe buffers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass

REPO_ID = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DATA_DIR = "sample/100BT"
DEFAULT_HOST = "frei178785@transfer1.bsc.es"
DEFAULT_TARGET = "/gpfs/scratch/ehpc390/fineweb-edu/sample-100BT-qwen3-budget30B"


@dataclass(frozen=True)
class Shard:
    path: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def url(self) -> str:
        path = urllib.parse.quote(self.path, safe="/")
        return (
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
            f"{REVISION}/{path}?download=true"
        )


def run_checked(command: list[str], *, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result.stdout.decode()


def load_manifest() -> list[Shard]:
    endpoint = (
        f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{REVISION}/"
        f"{DATA_DIR}?recursive=true&expand=true&limit=100"
    )
    payload = json.loads(
        run_checked(["curl", "--fail", "--silent", "--show-error", "--location", endpoint])
    )
    shards = []
    for entry in payload:
        path = entry.get("path", "")
        if entry.get("type") != "file" or not path.endswith(".parquet"):
            continue
        lfs = entry.get("lfs") or {}
        size = int(lfs.get("size", entry.get("size", 0)))
        sha256 = str(lfs.get("oid", ""))
        if size <= 0 or len(sha256) != 64:
            raise RuntimeError(f"Missing size or SHA-256 for {path}")
        shards.append(Shard(path=path, size=size, sha256=sha256))
    shards.sort(key=lambda shard: shard.path)
    if len(shards) != 100:
        raise RuntimeError(f"Expected 100 Parquet shards, found {len(shards)}")
    return shards


def remote_inventory(host: str, target: str) -> dict[str, int]:
    quoted_target = shlex.quote(target)
    command = (
        f"mkdir -p {quoted_target} && "
        f"find {quoted_target} -maxdepth 1 -type f "
        "\\( -name '*.parquet' -o -name '.*.parquet.part' \\) "
        "-printf '%f\\t%s\\n'"
    )
    output = run_checked(["ssh", "-o", "BatchMode=yes", host, command])
    inventory = {}
    for line in output.splitlines():
        name, size = line.split("\t", 1)
        inventory[name] = int(size)
    return inventory


def stream_shard(
    host: str,
    target: str,
    shard: Shard,
    verify_sha256: bool,
    resume_bytes: int,
) -> None:
    final_path = f"{target}/{shard.name}"
    partial_path = f"{target}/.{shard.name}.part"
    quoted_final = shlex.quote(final_path)
    quoted_partial = shlex.quote(partial_path)
    checks = [
        f"actual=$(stat -c %s {quoted_partial})",
        f'test "$actual" -eq {shard.size}',
    ]
    if verify_sha256:
        checks.extend(
            [
                f"actual_sha=$(sha256sum {quoted_partial})",
                'actual_sha=${actual_sha%% *}',
                f'test "$actual_sha" = {shlex.quote(shard.sha256)}',
            ]
        )
    setup = ["set -eu", "umask 002", f"mkdir -p {shlex.quote(target)}"]
    finish = [*checks, f"mv {quoted_partial} {quoted_final}"]
    if resume_bytes == shard.size:
        run_checked(["ssh", "-o", "BatchMode=yes", host, "; ".join([*setup, *finish])])
        return
    write_operator = ">>" if resume_bytes else ">"
    remote_command = "; ".join(
        [*setup, f"cat {write_operator} {quoted_partial}", *finish]
    )
    for attempt in range(1, 13):
        curl_command = [
            "curl",
            "--http1.1",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
        ]
        if resume_bytes:
            curl_command.extend(["--continue-at", str(resume_bytes)])
        curl_command.append(shard.url)
        curl = subprocess.Popen(curl_command, stdout=subprocess.PIPE)
        assert curl.stdout is not None
        ssh = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", host, remote_command],
            stdin=curl.stdout,
        )
        curl.stdout.close()
        ssh_status = ssh.wait()
        curl_status = curl.wait()
        if curl_status == 0 and ssh_status == 0:
            return

        status_command = (
            f"if test -f {quoted_final}; then printf 'final '; stat -c %s {quoted_final}; "
            f"elif test -f {quoted_partial}; then printf 'partial '; stat -c %s {quoted_partial}; "
            "else printf 'missing 0\\n'; fi"
        )
        for status_attempt in range(1, 4):
            try:
                status_output = run_checked(
                    ["ssh", "-o", "BatchMode=yes", host, status_command]
                )
                break
            except RuntimeError:
                if status_attempt == 3:
                    raise
                time.sleep(status_attempt)
        status, size_text = status_output.strip().split()
        new_resume_bytes = int(size_text)
        if status == "final" and new_resume_bytes == shard.size:
            return
        if status == "final" or new_resume_bytes > shard.size:
            raise RuntimeError(
                f"Transfer failed for {shard.name}: curl={curl_status}, ssh={ssh_status}, "
                f"remote partial={new_resume_bytes}, previous={resume_bytes}, expected={shard.size}"
            )
        if new_resume_bytes == shard.size:
            run_checked(["ssh", "-o", "BatchMode=yes", host, "; ".join([*setup, *finish])])
            return
        resume_bytes = new_resume_bytes
        write_operator = ">>" if resume_bytes else ">"
        remote_command = "; ".join(
            [*setup, f"cat {write_operator} {quoted_partial}", *finish]
        )
        if attempt == 12:
            break
        print(
            f"RETRY {shard.name} [{attempt}/12] at {resume_bytes / 1024**2:.0f} MiB",
            flush=True,
        )
        time.sleep(min(attempt, 5))

    raise RuntimeError(f"Transfer failed for {shard.name} after 12 attempts")


def write_remote_manifest(host: str, target: str, shards: list[Shard]) -> None:
    manifest = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "data_dir": DATA_DIR,
        "files": [shard.__dict__ for shard in shards],
        "total_bytes": sum(shard.size for shard in shards),
    }
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    partial = shlex.quote(f"{target}/MANIFEST.json.part")
    final = shlex.quote(f"{target}/MANIFEST.json")
    command = f"set -eu; cat > {partial}; mv {partial} {final}"
    run_checked(["ssh", "-o", "BatchMode=yes", host, command], input_bytes=payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("MN5_TRANSFER_HOST", DEFAULT_HOST))
    parser.add_argument("--target", default=os.environ.get("MN5_FINEWEB_DIR", DEFAULT_TARGET))
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent streams (default: 2)")
    parser.add_argument("--limit", type=int, help="Transfer a deterministic sample of N shards")
    parser.add_argument("--seed", type=int, default=42, help="Shard sampling seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Re-read each remote shard and verify its LFS SHA-256 before finalizing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    shards = load_manifest()
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit cannot be negative")
        if args.limit > len(shards):
            raise ValueError(f"--limit cannot exceed {len(shards)}")
        random.Random(args.seed).shuffle(shards)
        shards = sorted(shards[: args.limit], key=lambda shard: shard.path)
    total_gib = sum(shard.size for shard in shards) / 1024**3
    print(f"Official {REPO_ID}@{REVISION}")
    print(f"Destination: {args.host}:{args.target}")
    print(f"Selected: {len(shards)} shards, {total_gib:.1f} GiB")
    if args.dry_run:
        for shard in shards:
            print(f"  {shard.name}: {shard.size / 1024**3:.2f} GiB")
        return 0

    inventory = remote_inventory(args.host, args.target)
    pending = []
    for shard in shards:
        existing_size = inventory.get(shard.name)
        if existing_size is None:
            partial_size = inventory.get(f".{shard.name}.part", 0)
            if partial_size > shard.size:
                raise RuntimeError(
                    f"Partial {shard.name} has {partial_size} bytes; expected at most {shard.size}."
                )
            pending.append((shard, partial_size))
        elif existing_size == shard.size:
            print(f"SKIP {shard.name} (complete)")
        else:
            raise RuntimeError(
                f"Existing {shard.name} has {existing_size} bytes; expected {shard.size}. "
                "Move it aside before resuming."
            )

    print(f"Transferring {len(pending)} shard(s) with {args.jobs} concurrent stream(s)")
    lock = threading.Lock()
    completed = 0

    def transfer(item: tuple[Shard, int]) -> None:
        nonlocal completed
        shard, resume_bytes = item
        resume_note = f", resuming at {resume_bytes / 1024**2:.0f} MiB" if resume_bytes else ""
        print(
            f"START {shard.name} ({shard.size / 1024**3:.2f} GiB{resume_note})",
            flush=True,
        )
        stream_shard(args.host, args.target, shard, args.verify_sha256, resume_bytes)
        with lock:
            completed += 1
            print(f"DONE  {shard.name} [{completed}/{len(pending)}]", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(transfer, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    write_remote_manifest(args.host, args.target, shards)
    print("Transfer complete; remote MANIFEST.json written.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed shards remain valid and will be skipped next time.", file=sys.stderr)
        raise SystemExit(130)
