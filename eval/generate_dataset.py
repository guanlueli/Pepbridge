#!/usr/bin/env python3
"""
Generate peptide designs with PepBridge over a dataset of peptide-receptor
complexes, writing one PDB per design to:

    <output-root>/<complex_id>/intermediate_designs_inverse_folded/sample_{i}.pdb
    <output-root>/<complex_id>/intermediate_designs_inverse_folded/refold_cif/sample_{i}.pdb

The companion refold_cif/ directory is populated with the same designed
structures so downstream metrics that expect a "refold" subdir can run
immediately. Replace those files with the output of a separate refolder
(e.g. ESMFold) for a non-trivial designability comparison.

Inputs
------
* A pre-built PepDataset LMDB cache that contains surface features
  (surf_pos, surf_hp, surf_hbond, pts_rec_mask_symmetric) — see
  ``models_con/pep_dataloader.py``.
* The original per-complex peptide.pdb (under --dataset-root) is re-read at
  generation time to recover the native CA centre-of-mass. The LMDB stores
  peptide+receptor coords centred on this CoM; we add it back to generated
  heavy atoms before saving so output PDBs are in the native frame.

Sharding
--------
``--shard i --num-shards K`` selects every K-th complex starting at i.
Together with ``generate_dataset_3gpu.sh`` this gives embarrassingly-parallel
multi-GPU generation: each shard runs in its own process and writes to a
disjoint set of <complex_id>/ subdirs of the shared --output-root.

Example
-------
    python eval/generate_dataset.py \\
        --dataset-root  /path/to/peptide_testdata \\
        --lmdb-dir      /path/to/pepbridge_data \\
        --lmdb-name     pep_pocket_test_surf \\
        --checkpoint    /path/to/train_model1.pt \\
        --output-root   ./outputs/designs \\
        --num-samples 50 --num-steps 200 --mini-batch-size 10 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger("generate_dataset")

# Repo root = parent of eval/. With ``python eval/generate_dataset.py`` (run
# from anywhere), this resolves so models_con / pepbridge / data imports work.
PEPBRIDGE_REPO = Path(__file__).resolve().parent.parent


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _native_pep_center(pep_pdb_path: Path, parse_pdb, BBHeavyAtom) -> torch.Tensor | None:
    """Re-read the original peptide.pdb and return the CA CoM in native frame.

    PepBridge's LMDB stores peptide+receptor heavy atoms after subtracting this
    CoM, so we need it to un-centre generated peptides for downstream metrics.
    """
    if not pep_pdb_path.exists():
        return None
    try:
        pep = parse_pdb(str(pep_pdb_path))[0]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("parse_pdb failed for %s: %s", pep_pdb_path, exc)
        return None
    ca_mask = pep["mask_heavyatom"][:, BBHeavyAtom.CA]
    if int(ca_mask.sum().item()) == 0:
        return None
    center = torch.sum(
        pep["pos_heavyatom"][ca_mask, BBHeavyAtom.CA], dim=0
    ) / (ca_mask.sum().float() + 1e-8)
    return center


def _save_one_complex(
    *,
    item,
    center: torch.Tensor,
    num_samples: int,
    num_steps: int,
    mini_batch_size: int,
    sample_surf: bool,
    sample_bb: bool,
    sample_ang: bool,
    sample_seq: bool,
    model,
    device: str,
    out_design_dir: Path,
    out_refold_dir: Path | None,
    collate_fn,
    recursive_to,
    save_pdb,
    full_atom_reconstruction,
    get_heavyatom_mask,
) -> int:
    """Run PepBridge sample() for one complex in mini-batches and dump PDBs."""
    out_design_dir.mkdir(parents=True, exist_ok=True)
    if out_refold_dir is not None:
        out_refold_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    sample_idx = 0
    remaining = num_samples
    while remaining > 0:
        bs = min(mini_batch_size, remaining)
        data_list = [deepcopy(item) for _ in range(bs)]
        batch = recursive_to(collate_fn(data_list), device)

        with torch.no_grad():
            traj = model.sample(
                batch,
                num_steps=int(num_steps),
                sample_surf=bool(sample_surf),
                sample_bb=bool(sample_bb),
                sample_ang=bool(sample_ang),
                sample_seq=bool(sample_seq),
            )
        final = traj[-1]

        pos_ha, _, _ = full_atom_reconstruction(
            R_bb=final["rotmats"],
            t_bb=final["trans"],
            angles=final["angles"],
            aa=final["seqs"],
        )  # (B, L, 14, 3)
        # Pad heavy-atom dim from 14 to 15 to match downstream writer expectations.
        pos_ha = F.pad(pos_ha, pad=(0, 0, 0, 15 - 14), value=0.0)  # (B, L, 15, 3)

        batch_cpu = recursive_to(batch, "cpu")
        gen_mask = batch_cpu["generate_mask"]
        pos_ha = pos_ha.cpu()
        pos_new = torch.where(
            gen_mask[:, :, None, None], pos_ha, batch_cpu["pos_heavyatom"]
        )

        # Restore native frame uniformly across all residues.
        center_cpu = center.detach().cpu()
        pos_new = pos_new + center_cpu[None, None, None, :]

        mask_new = get_heavyatom_mask(final["seqs"]).cpu()
        mask_new = torch.where(
            gen_mask[:, :, None], mask_new, batch_cpu["mask_heavyatom"]
        )
        aa_new = final["seqs"].cpu()

        chain_id_first = list(zip(*batch_cpu["chain_id"]))[0] if batch_cpu["chain_id"] else []
        chain_id_first = list(chain_id_first)
        icode = [" " for _ in range(len(chain_id_first))]

        for i in range(bs):
            pdb_data = {
                "chain_nb": batch_cpu["chain_nb"][0],
                "chain_id": chain_id_first,
                "resseq": batch_cpu["resseq"][0],
                "icode": icode,
                "aa": aa_new[i],
                "mask_heavyatom": mask_new[i],
                "pos_heavyatom": pos_new[i],
            }
            design_path = out_design_dir / f"sample_{sample_idx:02d}.pdb"
            save_pdb(pdb_data, path=str(design_path))
            if out_refold_dir is not None:
                shutil.copyfile(design_path, out_refold_dir / f"sample_{sample_idx:02d}.pdb")
            sample_idx += 1
            saved += 1

        remaining -= bs
        torch.cuda.empty_cache()

    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pepbridge-repo",
        type=Path,
        default=PEPBRIDGE_REPO,
        help="Path to the Pepbridge repo (prepended to sys.path).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Per-complex PDB layout: <root>/<id>/peptide.pdb + pocket.pdb. Only "
            "peptide.pdb is re-read (to recover the native CA CoM for un-centring)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs/designs"),
        help="Where generated PDBs are written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pt checkpoint (model state dict under key 'model').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PEPBRIDGE_REPO / "configs" / "learn_surf_angle.yaml",
        help="YAML config that matches the checkpoint.",
    )
    parser.add_argument(
        "--lmdb-dir",
        type=Path,
        required=True,
        help="Directory containing <lmdb-name>_structure_cache.lmdb.",
    )
    parser.add_argument(
        "--lmdb-name",
        type=str,
        default="pep_pocket_test_surf",
        help="LMDB cache name prefix. File expected at <lmdb-dir>/<name>_structure_cache.lmdb.",
    )
    parser.add_argument(
        "--lmdb-structure-dir",
        type=Path,
        default=None,
        help=(
            "Directory PepDataset lists when (re)building the cache. Defaults to "
            "--dataset-root. Not re-read when the cache exists."
        ),
    )
    parser.add_argument(
        "--num-samples", type=int, default=4,
        help="Designs to generate per complex.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=1000,
        help="Bridge sampling steps (matches sampling.num_timesteps in the YAML).",
    )
    parser.add_argument(
        "--mini-batch-size", type=int, default=4,
        help="Max designs to run through model.sample() at once (caps GPU mem).",
    )

    def _b(s: str) -> bool:
        return s.lower() in {"1", "true", "yes", "y", "t"}

    parser.add_argument("--sample-surf", type=_b, default=True)
    parser.add_argument("--sample-bb", type=_b, default=True)
    parser.add_argument("--sample-ang", type=_b, default=True)
    parser.add_argument("--sample-seq", type=_b, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Only generate for these complex IDs (e.g. 1aze_B 1b07_C).",
    )
    parser.add_argument(
        "--only-from", type=Path, default=None,
        help="Path to a text file with complex IDs (one per line) to include.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N complexes.")
    parser.add_argument(
        "--shard", type=int, default=0,
        help=(
            "0-indexed shard ID. Together with --num-shards, processes only the "
            "complexes whose position satisfies i %% num_shards == shard."
        ),
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards. 1 means no sharding.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate even if the output design directory already has the expected pdbs.",
    )
    parser.add_argument(
        "--skip-refold-copy", action="store_true",
        help="Do NOT copy designs into refold_cif/. Useful when a separate refolder will run.",
    )
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    _setup_logging(args.log_level)

    if not args.checkpoint.exists():
        raise SystemExit(f"--checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"--config not found: {args.config}")
    lmdb_path = args.lmdb_dir / f"{args.lmdb_name}_structure_cache.lmdb"
    if not lmdb_path.exists():
        raise SystemExit(f"LMDB not found: {lmdb_path}")
    if not args.dataset_root.exists():
        raise SystemExit(f"--dataset-root not found: {args.dataset_root}")

    structure_dir = args.lmdb_structure_dir or args.dataset_root
    if not structure_dir.exists():
        raise SystemExit(
            f"--lmdb-structure-dir does not exist: {structure_dir}. "
            "PepDataset needs this to be a real directory even when the cache exists."
        )

    pepbridge_repo = args.pepbridge_repo.resolve()
    if str(pepbridge_repo) not in sys.path:
        sys.path.insert(0, str(pepbridge_repo))

    from pepbridge.utils.misc import load_config, seed_all
    from pepbridge.utils.train import recursive_to
    from pepbridge.utils.data import PaddingCollate
    from pepbridge.modules.protein.parsers import parse_pdb
    from pepbridge.modules.protein.writers import save_pdb
    from pepbridge.modules.protein.constants import BBHeavyAtom
    from models_con.diffusion_model import DiffusionModel
    from models_con.utils import process_dic
    from models_con.pep_dataloader import PepDataset
    from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask

    config, _ = load_config(str(args.config))
    seed_all(int(args.seed))

    LOGGER.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(str(args.checkpoint), map_location=args.device)
    model = DiffusionModel(config.model, args.device).to(args.device)
    model.load_state_dict(process_dic(ckpt["model"]))
    model.eval()

    collate_fn = PaddingCollate(eight=False)

    LOGGER.info(
        "Opening LMDB cache: %s  (structure_dir=%s)",
        lmdb_path, structure_dir,
    )
    dataset = PepDataset(
        structure_dir=str(structure_dir),
        dataset_dir=str(args.lmdb_dir),
        name=args.lmdb_name,
        transform=None,
        reset=False,
    )

    n_items = len(dataset)
    id_to_index = {dataset.db_ids[i]: i for i in range(n_items)}

    only_set: set[str] | None = None
    if args.only_from is not None:
        only_set = set()
        with args.only_from.open() as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    only_set.add(s)
    if args.only:
        if only_set is None:
            only_set = set()
        only_set.update(args.only)

    if only_set is not None:
        ordered_ids = [cid for cid in dataset.db_ids if cid in only_set]
    else:
        ordered_ids = list(dataset.db_ids)

    if args.limit is not None:
        ordered_ids = ordered_ids[: max(int(args.limit), 0)]

    if int(args.num_shards) < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= int(args.shard) < int(args.num_shards)):
        raise SystemExit(
            f"--shard must be in [0, {args.num_shards}), got {args.shard}"
        )
    if int(args.num_shards) > 1:
        all_ids = list(ordered_ids)
        ordered_ids = [
            cid for i, cid in enumerate(all_ids)
            if i % int(args.num_shards) == int(args.shard)
        ]
        LOGGER.info(
            "Sharded: shard %d / %d -> %d of %d complexes",
            int(args.shard), int(args.num_shards), len(ordered_ids), len(all_ids),
        )

    LOGGER.info(
        "Generating %d samples for %d complexes (of %d in LMDB) -> %s",
        int(args.num_samples), len(ordered_ids), n_items, args.output_root,
    )

    processed = 0
    skipped = 0
    failed = 0
    skip_reasons: dict[str, int] = {}
    failed_names: list[str] = []
    start = time.time()

    expected_files = {f"sample_{i:02d}.pdb" for i in range(int(args.num_samples))}

    for complex_id in ordered_ids:
        out_complex_dir = args.output_root / complex_id
        out_design_dir = out_complex_dir 
        out_refold_dir = None if args.skip_refold_copy else out_design_dir / "refold_cif"

        have_design = (
            out_design_dir.exists()
            and expected_files.issubset({p.name for p in out_design_dir.glob("sample_*.pdb")})
        )
        have_refold = (
            args.skip_refold_copy
            or (
                out_refold_dir is not None
                and out_refold_dir.exists()
                and expected_files.issubset({p.name for p in out_refold_dir.glob("sample_*.pdb")})
            )
        )
        if have_design and have_refold and not args.overwrite:
            skipped += 1
            skip_reasons["output_exists"] = skip_reasons.get("output_exists", 0) + 1
            LOGGER.info("[%s] skipped (outputs present)", complex_id)
            continue

        idx = id_to_index[complex_id]
        item = dataset[idx]

        pep_pdb = args.dataset_root / complex_id / "peptide.pdb"
        center = _native_pep_center(pep_pdb, parse_pdb, BBHeavyAtom)
        if center is None:
            skipped += 1
            skip_reasons["no_native_pep_pdb"] = skip_reasons.get("no_native_pep_pdb", 0) + 1
            LOGGER.info("[%s] skipped (cannot recover centre from %s)", complex_id, pep_pdb)
            continue

        t0 = time.time()
        try:
            n = _save_one_complex(
                item=item,
                center=center,
                num_samples=int(args.num_samples),
                num_steps=int(args.num_steps),
                mini_batch_size=int(args.mini_batch_size),
                sample_surf=args.sample_surf,
                sample_bb=args.sample_bb,
                sample_ang=args.sample_ang,
                sample_seq=args.sample_seq,
                model=model,
                device=args.device,
                out_design_dir=out_design_dir,
                out_refold_dir=out_refold_dir,
                collate_fn=collate_fn,
                recursive_to=recursive_to,
                save_pdb=save_pdb,
                full_atom_reconstruction=full_atom_reconstruction,
                get_heavyatom_mask=get_heavyatom_mask,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failed_names.append(complex_id)
            LOGGER.error(
                "[%s] FAILED: %s\n%s",
                complex_id, exc, traceback.format_exc(),
            )
            torch.cuda.empty_cache()
            continue

        processed += 1
        elapsed = time.time() - t0
        LOGGER.info("[%s] OK (%d pdbs) in %.1fs", complex_id, n, elapsed)
        torch.cuda.empty_cache()

    total_elapsed = time.time() - start
    LOGGER.info(
        "Done. processed=%d skipped=%d failed=%d elapsed=%.1fmin",
        processed, skipped, failed, total_elapsed / 60,
    )
    if skip_reasons:
        LOGGER.info("Skip reasons: %s", skip_reasons)
    if failed_names:
        LOGGER.info("Failed complexes: %s", failed_names)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
