import argparse
import pickle

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import rdkit.Chem as Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from configs.configs_data import data_configs

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import warnings

warnings.filterwarnings("ignore")


def calculate_energy(mol: Chem.Mol) -> float:
    """
    Returns: energy (NaN if failed)
    """
    mol = Chem.Mol(mol)  # Make a deep copy of the molecule
    mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = UFFGetMoleculeForceField(mol, confId=-1)
        return ff.CalcEnergy()
    except Exception:
        return np.nan


def relax_global(mol: Chem.Mol) -> Chem.Mol:
    """Relax a molecule in vaccum by the UFF force field.
    Returns:
        Chem.Mol: The relaxed molecule.
    """
    if mol is None:
        return None
    Chem.GetSSSR(mol)  # SSSR: Smallest Set of Smallest Rings

    mol = deepcopy(mol)
    mol = Chem.AddHs(mol, addCoords=True)

    try:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return None
    return mol


def relax_global_on_pose(mol: Chem.Mol) -> Chem.Mol:
    """Relax the GIVEN pose WITHOUT CONSTRAINS
    Returns:
        Chem.Mol: The relaxed molecule.
    """
    if mol is None:
        return None

    Chem.GetSSSR(mol)
    mol = deepcopy(mol)
    mol = Chem.AddHs(mol, addCoords=True)

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return None
    return mol


def get_global_min_energy_worker(args):
    """
    Takes a tuple (ccd_id, mol) and runs the exact global energy logic
    extracted from your calculate_strain_energy function.
    """
    idx, (ccd_id, mol), num_confs = args

    if mol is None:
        return ccd_id, np.nan

    try:
        global_relaxed = [relax_global(mol) for _ in range(num_confs)]
        global_relaxed.append(relax_global_on_pose(mol))

        energies = [calculate_energy(m) for m in global_relaxed if m is not None]
        valid_energies = [e for e in energies if not np.isnan(e)]

        if not valid_energies:
            return ccd_id, np.nan

        return ccd_id, min(valid_energies)

    except Exception:
        return ccd_id, np.nan


def main():

    parser = argparse.ArgumentParser(
        description="Compute UFF global minimum energies for RDKit molecules."
    )
    parser.add_argument(
        "--input_dir",
        default=data_configs["ccd_components_rdkit_mol_dir"],
        help="Directory of per-CCD pickles, one {CCD}.pkl -> {CCD: mol} per component",
    )
    parser.add_argument(
        "--output",
        default=data_configs["ccd_components_mol_energy_file"],
        help="Output pickle file path {ccd_id: energy}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute every molecule instead of only those missing from --output",
    )
    parser.add_argument(
        "--num-confs",
        type=int,
        default=50,
        help="Number of random global conformations",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of worker threads",
    )

    args = parser.parse_args()

    results_dict = {}
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "rb") as f:
            results_dict = pickle.load(f)
        print(f"Loaded {len(results_dict):,} existing energies from {args.output}.")

    pkl_files = [
        p
        for p in sorted(Path(args.input_dir).glob("*.pkl"))
        if p.stem not in results_dict
    ]
    print(f"Loading {len(pkl_files):,} new molecules from {args.input_dir}...")

    tasks = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            tasks.append((pkl_file.stem, pickle.load(f)[pkl_file.stem]))

    print(f"Processing {len(tasks)} molecules with {args.threads} threads...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                get_global_min_energy_worker, (i, task, args.num_confs)
            ): task[0]
            for i, task in enumerate(tasks)
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                ccd_id, energy = future.result()
                results_dict[ccd_id] = energy
            except Exception as e:
                failed_id = futures[future]
                print(f"Error processing {failed_id}: {e}")
                results_dict[failed_id] = np.nan

    print(f"Saving results to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(results_dict, f)

    print("Done!")


if __name__ == "__main__":
    main()
