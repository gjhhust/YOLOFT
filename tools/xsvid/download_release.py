#!/usr/bin/env python3
"""Anonymous HF/MS release download, integrity verification and legacy preparation.

Standard library only. No hub SDK, token, cookie jar or credential file is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PAPER_SOT_SHA256 = "c59639f868a5fdfa5474f072d657bafb072430c7386b12901e3bc1ac7a42188a"
TRANST_PRETRAIN = json.loads(Path(__file__).with_name("transt_pretrain_manifest.json").read_text())
MODEL_HASHES = {TRANST_PRETRAIN['path']: TRANST_PRETRAIN['sha256'], 'checkpoints/reid/osnet_x0_25_msmt17.pt': '6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99', 'configs/yoloft-l-temporal.yaml': '114c4b22929a35314a84eeb3d5252d9d791061213100e7c70e8a58857c60ecc2', 'protocols/paper_motchallenge_gt.zip': 'c00b0e091902af2add831339867f05e52ddab5dfab7ccb9e00b357100605e34b'}
MODEL_METADATA_HASHES = {
    "protocols/paper_motchallenge_mapping.json": "47ee880cae4b20ad51a1106b653e591e10cf9c18da57cbeb95ff8ed49219a51c",
}
RELEASE_MANIFEST = json.loads(Path(__file__).with_name("release_manifest.json").read_text())
RELEASE_HASHES = {
    **{name: record["file_sha256"] for name, record in RELEASE_MANIFEST["weights"].items()},
}
DATASET_HASHES = {
    "archives/manifest.json": "e340e7f5b5656d8d7f07b609fd3f95222924cbf410353409424971a3596cd19a",
    "tools/prepare_yolo_labels.py": "5dfe4669fd69c8e5bf31b4b9941728245285e7c2164309044b415db4e030c695",
    "tools/build_unified_annotations.py": "b56536860686e5672bc19bb8eefe8488c34deeb86b80e92265cf0f7fddcad22d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal, Windows paths and existing symlink components."""
    parts = PurePosixPath(name).parts
    if not parts or name.startswith("/") or "\\" in name or ":" in name or ".." in parts:
        raise ValueError(f"Unsafe release path: {name!r}")
    root = Path(os.path.abspath(root))
    target = root
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"Symlink destination: {target}")
    if root.is_symlink() or root.resolve() != root.absolute():
        raise ValueError(f"Symlink root/ancestor: {root}")
    return target


def verify(path: Path, expected: str, size: int = None) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid SHA-256 for {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: expected {expected}, got {actual}")


def fetch(url: str, target: Path, expected: str = None, size: int = None) -> None:
    if expected and target.is_file():
        verify(target, expected, size)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "XSVID-anonymous-download/1"})
            with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".download-", delete=False
            ) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output, 1024 * 1024)
            if expected:
                verify(temporary, expected, size)
            elif size is not None and temporary.stat().st_size != size:
                raise ValueError(f"Size mismatch: {target}")
            os.replace(temporary, target)
            return
        except (OSError, ValueError):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def read_json(url: str):
    with urlopen(Request(url, headers={"User-Agent": "XSVID-anonymous-download/1"}), timeout=60) as response:
        return json.load(response), response.headers


def ms_success(payload: dict) -> bool:
    if "Success" in payload:
        return payload["Success"] is True
    return payload.get("Code") in (200, "200")


def list_files(hub: str, repo: str, kind: str, revision: str):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError(f"Invalid repo ID: {repo}")
    encoded = quote(revision, safe="")
    if hub == "hf":
        url = f"https://huggingface.co/api/{kind}s/{repo}/tree/{encoded}?recursive=true&limit=1000"
        while url:
            records, headers = read_json(url)
            if not isinstance(records, list):
                raise ValueError(f"HF anonymous listing failed: {records}")
            for record in records:
                if record["type"] == "file":
                    yield record["path"], record.get("lfs", {}).get("oid"), record.get("size")
            link = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", ""))
            url = link.group(1) if link else None
            if url and urlparse(url).netloc != "huggingface.co":
                raise ValueError("Unsafe HF pagination link")
    else:
        base = "https://modelscope.cn/api/v1"
        if kind == "dataset":
            metadata, _ = read_json(f"{base}/datasets/{repo}")
            if not ms_success(metadata):
                raise ValueError(f"MS anonymous dataset lookup failed: {metadata}")
            endpoint = f"{base}/datasets/{metadata['Data']['Id']}/repo/tree"
        else:
            endpoint = f"{base}/models/{repo}/repo/files"
        page = 1
        while True:
            params = {"Revision": revision, "Recursive": "True", "Root": "/", "PageNumber": page, "PageSize": 100}
            payload, _ = read_json(endpoint + "?" + urlencode(params))
            if not ms_success(payload):
                raise ValueError(f"MS anonymous listing failed: {payload}")
            records = payload["Data"]["Files"]
            for record in records:
                if record["Type"] not in ("tree", "dir"):
                    yield record["Path"], record.get("Sha256"), record.get("Size")
            if kind == "model" or len(records) < 100:
                break
            page += 1


def file_url(hub: str, repo: str, kind: str, revision: str, name: str) -> str:
    if hub == "hf":
        prefix = "datasets/" if kind == "dataset" else ""
        return f"https://huggingface.co/{prefix}{repo}/resolve/{quote(revision, safe='')}/{quote(name)}"
    return f"https://modelscope.cn/api/v1/{kind}s/{repo}/repo?" + urlencode(
        {"Revision": revision, "FilePath": name, "Source": "SDK", "View": "False"}
    )


def snapshot(hub: str, repo: str, kind: str, revision: str, root: Path, paper_sot: Path = None,
             model_hashes: dict = None) -> None:
    model_hashes = MODEL_HASHES if model_hashes is None else model_hashes
    for name, digest, size in list_files(hub, repo, kind, revision):
        if name == ".gitattributes" or name.startswith("."):
            continue
        parts = PurePosixPath(name).parts
        if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
            continue
        if kind == "model":
            basename = parts[-1].lower()
            metadata = (len(parts) == 1 and basename.startswith(("readme", "license"))) or (
                name.endswith(".json") and ("manifest" in basename or "metrics" in basename
                                           or parts[0] in ("metrics", "protocols")))
            if parts[0] == "code" or (name not in model_hashes and not metadata):
                continue  # Source comes from this checkout, not the model snapshot.
        if kind == "model" and name == TRANST_PRETRAIN["source_path"]:
            continue  # Original ltr pickle is provenance, not portable training input.
        if kind == "dataset" and (name.startswith("images/") or name.endswith(".tar")):
            continue  # Media are fetched from the archive manifest after metadata.
        if paper_sot and name == "checkpoints/yoloft-l-xsvid-v2-sot.pt":
            continue
        expected = (model_hashes if kind == "model" else DATASET_HASHES).get(
            name, MODEL_METADATA_HASHES.get(name, digest) if kind == "model" else digest)
        if kind == "model" and name == TRANST_PRETRAIN["path"]:
            size = TRANST_PRETRAIN["size_bytes"]
        print(f"Download {kind}: {name}", flush=True)
        fetch(file_url(hub, repo, kind, revision, name), safe_path(root, name), expected, size)


def safe_extract(archive: Path, root: Path) -> None:
    """Validate every member before writing; never follow archive links."""
    with tarfile.open(archive, "r:") as stream:
        members = stream.getmembers()
        seen = set()
        total = 0
        for member in members:
            target = safe_path(root, member.name)
            if PurePosixPath(member.name).parts[0] != "images" or not (member.isfile() or member.isdir()):
                raise ValueError(f"Disallowed tar member: {member.name}")
            if target in seen or member.size < 0 or member.issparse():
                raise ValueError(f"Duplicate/sparse/invalid tar member: {member.name}")
            seen.add(target)
            total += member.size
        if total > archive.stat().st_size:
            raise ValueError("Tar expansion exceeds uncompressed archive size")
        for member in members:
            target = safe_path(root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with stream.extractfile(member) as source, tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix=".extract-", delete=False
                    ) as output:
                        temporary = Path(output.name)
                        shutil.copyfileobj(source, output, 1024 * 1024)
                    os.replace(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)


def prepare_dataset(root: Path, hub: str, repo: str, revision: str, offline: bool, protocol: str) -> None:
    for name, expected in DATASET_HASHES.items():
        verify(safe_path(root, name), expected)
    manifest = json.loads(safe_path(root, "archives/manifest.json").read_text())
    shards = manifest["shards"]
    if not shards or len({s["path"] for s in shards}) != len(shards):
        raise ValueError("Missing/duplicate archive shards")
    for shard in shards:
        name = "archives/" + shard["path"]
        archive = safe_path(root, name)
        if not offline:
            fetch(file_url(hub, repo, "dataset", revision, name), archive, shard["sha256"], shard["bytes"])
        verify(archive, shard["sha256"], shard["bytes"])
        print(f"Verified; extracting {archive.name}", flush=True)
        safe_extract(archive, root)
    subprocess.run([sys.executable, str(ROOT / "tools/xsvid/prepare_legacy_layout.py"),
                    "--data-root", str(root), "--detection-protocol", protocol, "--regenerate-labels"], check=True)


def prepare_model(root: Path, paper_sot: Path = None, *, current_release: bool = True) -> None:
    if paper_sot:
        raise ValueError("Only the current model package is supported")
    for name, expected in RELEASE_HASHES.items():
        size = TRANST_PRETRAIN["size_bytes"] if name == TRANST_PRETRAIN["path"] else None
        verify(safe_path(root, name), expected, size)
    for name, expected in MODEL_METADATA_HASHES.items():
        path = safe_path(root, name)
        if path.exists():
            verify(path, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", choices=("hf", "ms"), default="hf")
    parser.add_argument("--component", choices=("all", "dataset", "model"), default="dataset",
                        help="Default dataset; public model package is on HF only")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dataset-repo")
    parser.add_argument("--model-repo")
    parser.add_argument("--model-profile", choices=("current",), default="current",
                        help="YOLOFT release version and strict package hashes")
    parser.add_argument("--revision", help="Default main (HF) or master (MS); pin a commit for immutable downloads")
    parser.add_argument("--offline", action="store_true", help="Verify and prepare already downloaded files")
    parser.add_argument("--paper-sot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--detection-protocol", choices=("paper", "canonical"), default="paper")
    args = parser.parse_args()
    if args.paper_sot:
        parser.error("The current release does not support --paper-sot substitution")
    if not args.offline and args.hub == "ms" and args.component in ("all", "model") and not args.model_repo:
        parser.error("The public MS mirror contains the dataset only. Use --hub hf --component model "
                     "for weights, or supply an independently verified --model-repo explicitly.")
    destination = args.destination.absolute()
    revision = args.revision or ("main" if args.hub == "hf" else "master")
    owner = "lanlanlan23" if args.hub == "hf" else "lanlanlanrr"
    try:
        for kind, folder, repo in (
            ("dataset", "XS-VID-v2", args.dataset_repo or f"{owner}/XS-VID-v2"),
            ("model", "YOLOFT-XSVID-v2-weights", args.model_repo or RELEASE_MANIFEST["model_repository"]),
        ):
            if args.component not in ("all", kind):
                continue
            root = safe_path(destination, folder)
            root.mkdir(parents=True, exist_ok=True)
            if not args.offline:
                if kind == "model":
                    snapshot(args.hub, repo, kind, revision, root, model_hashes=RELEASE_HASHES)
                else:
                    snapshot(args.hub, repo, kind, revision, root, args.paper_sot)
            if kind == "dataset":
                prepare_dataset(root, args.hub, repo, revision, args.offline, args.detection_protocol)
            else:
                prepare_model(root, current_release=True)
        print("RELEASE_PREPARED (integrity/preparation only, not GPU reproduction)")
    except (OSError, ValueError, KeyError, tarfile.TarError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release preparation failed: {exc}\n"
                    "Check anonymous repository access, the selected revision and package file SHA-256 values. "
                    "The model package contains vid.pt, unified_mot.pt and unified_sot.pt.\n")


if __name__ == "__main__":
    main()
