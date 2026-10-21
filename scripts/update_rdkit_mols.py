import os
import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from configs.configs_data import data_configs


def get_fp_binary(mol):
    try:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol)
        return fp.ToBinary()
    except Exception:
        return None


def make_ccd_mol_from_rdkit_mol(rdkit_mol: Chem.Mol, ccd_code: str) -> Chem.Mol:
    index = 1
    for atom in rdkit_mol.GetAtoms():
        elem = atom.GetSymbol().upper()
        atom_name = f"{elem}{index}"
        atom.SetProp("name", atom_name)
        index += 1

    rdkit_mol.atom_map = {
        atom.GetProp("name"): atom.GetIdx() for atom in rdkit_mol.GetAtoms()
    }
    rdkit_mol.ref_mask = np.ones(len(rdkit_mol.GetAtoms()), dtype=bool)
    rdkit_mol.ref_conf_id = 0
    rdkit_mol.name = ccd_code
    return rdkit_mol


def load_existing_fps(rdkit_mol_dir: Path) -> set:
    existing_fps = set()
    pkl_files = list(rdkit_mol_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files):,} existing CCD molecules in {rdkit_mol_dir}.")

    for pkl_file in tqdm(pkl_files, desc="Indexing existing molecules (Fingerprints)"):
        try:
            with open(pkl_file, "rb") as f:
                mol = pickle.load(f)[pkl_file.stem]
            mol_no_h = Chem.RemoveAllHs(mol)
            fp = get_fp_binary(mol_no_h)
            if fp:
                existing_fps.add(fp)
        except Exception:
            continue

    return existing_fps


def update_components_with_ph(sdf_dir, rdkit_mol_dir):
    rdkit_mol_dir = Path(rdkit_mol_dir)
    rdkit_mol_dir.mkdir(parents=True, exist_ok=True)
    existing_fps = load_existing_fps(rdkit_mol_dir)

    print(f"\nScanning ligands in: {sdf_dir}")
    added_count = 0

    for fname in tqdm(os.listdir(sdf_dir), desc="Processing SDFs"):
        if not fname.endswith(".sdf"):
            continue

        # Upper-case to match res_name lookups and update_components_cif.py.
        sdf_code = os.path.splitext(fname)[0].upper()
        sdf_path = os.path.join(sdf_dir, fname)
        out_pkl = rdkit_mol_dir / f"{sdf_code}.pkl"

        if out_pkl.exists():
            continue

        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        try:
            mols = [m for m in suppl if m is not None]
        except:
            continue

        if not mols:
            continue

        rd = make_ccd_mol_from_rdkit_mol(mols[0], ccd_code=sdf_code)

        try:
            rd_no_h = Chem.RemoveAllHs(rd)
            fp_new = get_fp_binary(rd_no_h)
        except Exception:
            continue

        if fp_new is None or fp_new in existing_fps:
            continue

        with open(out_pkl, "wb") as f:
            pickle.dump({sdf_code: rd}, f)

        existing_fps.add(fp_new)
        added_count += 1

    print("\n------------------------------")
    print(f"Added {added_count} new molecules.")
    print(f"Total molecules now: {len(list(rdkit_mol_dir.glob('*.pkl'))):,}")
    print(f"Per-CCD pickles saved → {rdkit_mol_dir}")
    print("------------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Add new ligands to the per-CCD rdkit mol directory (ccd_rdkit_mol/)"
    )

    parser.add_argument("--sdf_dir", required=True, help="Path to ligand SDF folder")
    parser.add_argument(
        "--rdkit_mol_dir",
        default=data_configs["ccd_components_rdkit_mol_dir"],
        help="Directory of per-CCD pickles, one {CCD}.pkl -> {CCD: mol} per component",
    )

    args = parser.parse_args()

    update_components_with_ph(
        sdf_dir=args.sdf_dir,
        rdkit_mol_dir=args.rdkit_mol_dir,
    )


if __name__ == "__main__":
    main()
