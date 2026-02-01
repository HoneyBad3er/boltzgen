from torch import Tensor
from typing import Dict
import torch

from boltzgen.data.mol import minimum_lddt_symmetry_coords
from boltzgen.data.pad import pad_dim
from boltzgen.model.loss.diffusion import weighted_rigid_align
from boltzgen.model.loss.validation import weighted_minimum_rmsd


def minimum_rmsd_symmetry_coords(
    coords: Tensor,
    feats: Dict[str, Tensor],
    index_batch: int,
):
    all_coords = feats["all_coords"][index_batch].unsqueeze(0).to(coords)
    all_resolved_mask = (
        feats["all_resolved_mask"][index_batch].to(coords).to(torch.bool)
    )
    crop_to_all_atom_map = (
        feats["crop_to_all_atom_map"][index_batch].to(coords).to(torch.long)
    )
    chain_swaps = feats["chain_swaps"][index_batch]

    pred_coords = coords[:, : len(crop_to_all_atom_map)]

    # Default: no swap
    best_true_coords = all_coords[:, crop_to_all_atom_map].clone()
    best_true_resolved_mask = all_resolved_mask[crop_to_all_atom_map].clone()
    best_rmsd = float("inf")

    for c in chain_swaps:
        true_all_coords = all_coords.clone()
        true_all_resolved_mask = all_resolved_mask.clone()
        for start1, end1, start2, end2, chainidx1, chainidx2 in c:
            true_all_coords[:, start1:end1] = all_coords[:, start2:end2]
            true_all_resolved_mask[start1:end1] = all_resolved_mask[start2:end2]

        true_coords = true_all_coords[:, crop_to_all_atom_map]
        true_resolved_mask = true_all_resolved_mask[crop_to_all_atom_map]
        mask = true_resolved_mask.unsqueeze(0)

        if torch.sum(true_resolved_mask) <= 3:
            continue

        weights = torch.ones_like(mask, dtype=coords.dtype)
        aligned_true = weighted_rigid_align(
            true_coords, pred_coords, weights, mask=mask
        )
        mse = ((pred_coords - aligned_true) ** 2).sum(dim=-1)
        denom = (weights * mask).sum(dim=-1).clamp_min(1e-7)
        rmsd = torch.sqrt(torch.sum(mse * weights * mask, dim=-1) / denom)

        if rmsd.item() < best_rmsd:
            best_rmsd = rmsd.item()
            best_true_coords = true_coords
            best_true_resolved_mask = true_resolved_mask

    # Pad to match coords length
    best_true_coords = pad_dim(
        best_true_coords, 1, coords.shape[1] - best_true_coords.shape[1]
    )
    best_true_resolved_mask = pad_dim(
        best_true_resolved_mask, 0, coords.shape[1] - best_true_resolved_mask.shape[0]
    )

    return best_true_coords, best_true_resolved_mask.unsqueeze(0)


def get_true_coordinates(
    gt_data: Dict[str, Tensor],
    predictions: Dict[str, Tensor],
    n_diffusion_samples: int,
    symmetry_correction: bool,  
    expand_to_diffusion_samples: bool = True,
    protein_lig_rmsd=False,
    rmsd_symmetry_correction: bool = False,
):
    if symmetry_correction:
        msg = "expand_to_diffusion_samples must be true for symmetry correction."
        assert expand_to_diffusion_samples, msg

    return_dict = {}

    if (
        symmetry_correction
    ):
        K = gt_data["coords"].shape[1]
        assert K == 1, (
            f"Symmetry correction is not supported for num_ensembles_val={K}."
        )

        assert gt_data["coords"].shape[0] == 1, (
            f"Validation is not supported for batch sizes={gt_data['coords'].shape[0]}"
        )

        true_coords = []
        true_coords_resolved_mask = []
        for idx in range(gt_data["token_index"].shape[0]):
            for rep in range(n_diffusion_samples):
                i = idx * n_diffusion_samples + rep

                best_true_coords, best_true_coords_resolved_mask = (
                    minimum_lddt_symmetry_coords(
                        coords=predictions["sample_atom_coords"][i : i + 1],
                        feats=gt_data,
                        index_batch=idx,
                    )
                )

                true_coords.append(best_true_coords)
                true_coords_resolved_mask.append(best_true_coords_resolved_mask)

        true_coords = torch.cat(true_coords, dim=0)
        true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
        true_coords = true_coords.unsqueeze(1)


        return_dict["true_coords"] = true_coords
        return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
        return_dict["rmsds"] = 0
        return_dict["best_rmsd_recall"] = 0

    else:
        assert gt_data["coords"].shape[0] == 1, (
            f"Validation is not supported for batch sizes={gt_data['coords'].shape[0]}"
        )
        K, L = gt_data["coords"].shape[1:3]

        true_coords_resolved_mask = gt_data["atom_resolved_mask"] 
        true_coords = gt_data["coords"].squeeze(0)
        if expand_to_diffusion_samples:
            true_coords = true_coords.repeat((n_diffusion_samples, 1, 1)).reshape(
                n_diffusion_samples, K, L, 3
            )

            true_coords_resolved_mask = true_coords_resolved_mask.repeat_interleave(
                n_diffusion_samples, dim=0
            )  # since all masks are the same across conformers and diffusion samples, can just repeat S times
        else:
            true_coords_resolved_mask = true_coords_resolved_mask.squeeze(0)

        return_dict["true_coords"] = true_coords
        return_dict["true_coords_resolved_mask"] = true_coords_resolved_mask
        return_dict["rmsds"] = 0
        return_dict["best_rmsd_recall"] = 0
        return_dict["best_rmsd_precision"] = 0

        if protein_lig_rmsd:
            rmsd_batch = gt_data
            if rmsd_symmetry_correction:
                if "chain_swaps" not in gt_data:
                    print(
                        "Warning: rmsd_symmetry_correction requested but symmetry features are missing."
                    )
                elif n_diffusion_samples != 1 or gt_data["coords"].shape[1] != 1:
                    print(
                        "Warning: rmsd_symmetry_correction currently supports diffusion_samples=1 and K=1. Falling back to standard RMSD."
                    )
                else:
                    best_true_coords, best_true_coords_resolved_mask = (
                        minimum_rmsd_symmetry_coords(
                            coords=predictions["sample_atom_coords"][0:1],
                            feats=gt_data,
                            index_batch=0,
                        )
                    )
                    rmsd_batch = dict(gt_data)
                    rmsd_batch["coords"] = best_true_coords.unsqueeze(1)
                    rmsd_batch[
                        "atom_resolved_mask"
                    ] = best_true_coords_resolved_mask

            (
                rmsd,
                best_rmsd,
                rmsd_design,
                best_rmsd_design,
                rmsd_target,
                best_rmsd_target,
                rmsd_design_target,
                best_rmsd_design_target,
                target_aligned_rmsd_design,
                best_target_aligned_rmsd_design,
            ) = weighted_minimum_rmsd(
                predictions["sample_atom_coords"],
                rmsd_batch,
                multiplicity=n_diffusion_samples,
                protein_lig_rmsd=protein_lig_rmsd,
            )
            return_dict["rmsd"] = rmsd
            return_dict["best_rmsd"] = best_rmsd
            return_dict["rmsd_design"] = rmsd_design
            return_dict["best_rmsd_design"] = best_rmsd_design
            return_dict["rmsd_target"] = rmsd_target
            return_dict["best_rmsd_target"] = best_rmsd_target
            return_dict["rmsd_design_target"] = rmsd_design_target
            return_dict["best_rmsd_design_target"] = best_rmsd_design_target
            return_dict["target_aligned_rmsd_design"] = target_aligned_rmsd_design
            return_dict["best_target_aligned_rmsd_design"] = best_target_aligned_rmsd_design
    return return_dict
