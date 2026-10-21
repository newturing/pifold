import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import tqdm

from configs.configs_data import data_configs


def create_cif_block(mol: Chem.Mol, ccd_code: str) -> str:
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ccd_code

    cif_lines = [
        f"data_{ccd_code}",
        "#",
        "_chem_comp.id                                    " + ccd_code,
        f"_chem_comp.name                                  '{name}'",
        "_chem_comp.type                                  'non-polymer'",
        "_chem_comp.pdbx_type                             'HETIC'",
        f"_chem_comp.formula                               '{formula}'",
        f"_chem_comp.mon_nstd_parent_comp_id               ?",
        f"_chem_comp.pdbx_formal_charge                    {Chem.GetFormalCharge(mol)}",
        f"_chem_comp.pdbx_initial_date                     ?",
        f"_chem_comp.pdbx_modified_date                    ?",
        f"_chem_comp.pdbx_ambiguous_flag                   ?",
        f"_chem_comp.pdbx_release_status                   'REL'",
        f"_chem_comp.pdbx_replaced_by                      ?",
        f"_chem_comp.pdbx_replaces                         ?",
        f"_chem_comp.formula_weight                        {mol_weight:.3f}",
        f"_chem_comp.one_letter_code                       ?",
        f"_chem_comp.three_letter_code                     " + ccd_code,
        f"_chem_comp.pdbx_model_coordinates_details        ?",
        f"_chem_comp.pdbx_model_coordinates_missing_flag   N",
        f"_chem_comp.pdbx_ideal_coordinates_details        Corina",
        f"_chem_comp.pdbx_ideal_coordinates_missing_flag   N",
        f"_chem_comp.pdbx_subcomponent_list                ?",
        f"_chem_comp.pdbx_processing_site                  RCSB",
        "#",
    ]

    cif_lines.extend(
        [
            "loop_",
            "_chem_comp_atom.comp_id",
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.alt_atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_align",
            "_chem_comp_atom.pdbx_aromatic_flag",
            "_chem_comp_atom.pdbx_leaving_atom_flag",
            "_chem_comp_atom.pdbx_stereo_config",
            "_chem_comp_atom.model_Cartn_x",
            "_chem_comp_atom.model_Cartn_y",
            "_chem_comp_atom.model_Cartn_z",
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
            "_chem_comp_atom.pdbx_component_atom_id",
            "_chem_comp_atom.pdbx_component_comp_id",
            "_chem_comp_atom.pdbx_ordinal",
        ]
    )

    conformer = mol.GetConformer()
    atom_names = []

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conformer.GetAtomPosition(i)

        if atom.GetPDBResidueInfo() and atom.GetPDBResidueInfo().GetName():
            atom_name = atom.GetPDBResidueInfo().GetName().strip().upper()
        else:
            atom_name = f"{atom.GetSymbol().upper()}{atom.GetIdx() + 1}"

        atom_names.append(atom_name)
        atom_symbol = atom.GetSymbol().upper()
        formal_charge = atom.GetFormalCharge() if atom.GetFormalCharge() != 0 else "?"
        is_aromatic = "Y" if atom.GetIsAromatic() else "N"

        row = (
            f"{ccd_code:<10} {atom_name:<9} {atom_name:<11} {atom_symbol:<11} {formal_charge:<6} 1 "
            f"{is_aromatic:<2} N  N    {pos.x:8.3f} {pos.y:8.3f} {pos.z:8.3f} "
            f"{pos.x:8.3f} {pos.y:8.3f} {pos.z:8.3f} {atom_name:<24} "
            f"{ccd_code:<24} {i + 1}"
        )
        cif_lines.append(row)

    cif_lines.append("#")
    cif_lines.extend(
        [
            "loop_",
            "_chem_comp_bond.comp_id",
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
            "_chem_comp_bond.pdbx_aromatic_flag",
            "_chem_comp_bond.pdbx_stereo_config",
            "_chem_comp_bond.pdbx_ordinal",
        ]
    )

    bond_order_map = {
        Chem.BondType.SINGLE: "SING",
        Chem.BondType.DOUBLE: "DOUB",
        Chem.BondType.TRIPLE: "TRIP",
    }

    mol_copy = Chem.Mol(mol)
    Chem.Kekulize(mol_copy, clearAromaticFlags=False)

    for idx, bond in enumerate(mol.GetBonds(), start=1):
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_copy = mol_copy.GetBondBetweenAtoms(a1, a2)
        bond_type = bond_order_map.get(bond_copy.GetBondType(), "SING")
        aromatic = "Y" if bond.GetIsAromatic() else "N"

        cif_lines.append(
            f"{ccd_code:<10} {atom_names[a1]:<8} {atom_names[a2]:<8} "
            f"{bond_type:<4} {aromatic:<2} N {idx}"
        )

    cif_lines.append("#")
    return "\n".join(cif_lines)


def write_sdfs_to_ccd_dir(
    sdf_dir: Path,
    ccd_cif_dir: Path,
    rdkit_mol_dir: Path,
):
    sdf_files = list(sdf_dir.glob("*.sdf"))

    if not sdf_files:
        print(f"No SDF files found in {sdf_dir}.")
        return

    ccd_cif_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(sdf_files)} SDF files...")
    count = 0

    for sdf_file in tqdm.tqdm(sdf_files, desc="Converting SDFs"):
        ccd_code = sdf_file.stem.upper()
        out_cif = ccd_cif_dir / f"{ccd_code}.cif"

        # Only ligands registered by update_rdkit_mols.py (not duplicates of existing CCDs).
        if not (rdkit_mol_dir / f"{ccd_code}.pkl").exists():
            continue
        if out_cif.exists():
            continue

        try:
            suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=True)
            mol = next(suppl)

            if mol is None:
                print(f"Failed to read molecule: {sdf_file}")
                continue

            if mol.GetNumConformers() == 0:
                print(f"{ccd_code} has no 3D coords. Skipping.")
                continue

            cif_block = create_cif_block(mol, ccd_code)
            out_cif.write_text(cif_block + "\n")
            count += 1

        except Exception as e:
            print(f"Error processing {sdf_file}: {e}")

    print(f"Written {count} CIF files to {ccd_cif_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Write SDF ligands as per-CCD CIF files (ccd_cif/)."
    )
    parser.add_argument(
        "--sdf_dir", required=True, help="Directory containing SDF files"
    )
    parser.add_argument(
        "--ccd_cif_dir",
        default=data_configs["ccd_components_dir"],
        help="Directory of per-CCD CIF files, one {CCD}.cif per component",
    )
    parser.add_argument(
        "--rdkit_mol_dir",
        default=data_configs["ccd_components_rdkit_mol_dir"],
        help="Directory of per-CCD pickles written by update_rdkit_mols.py",
    )

    args = parser.parse_args()

    write_sdfs_to_ccd_dir(
        sdf_dir=Path(args.sdf_dir),
        ccd_cif_dir=Path(args.ccd_cif_dir),
        rdkit_mol_dir=Path(args.rdkit_mol_dir),
    )


if __name__ == "__main__":
    main()
