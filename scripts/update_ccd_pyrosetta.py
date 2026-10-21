import os
import glob
import argparse

from configs.configs_data import data_configs

COMPONENTS_DIR = data_configs["ccd_components_dir"]


def split_components(components_dir: str, lib_dir: str):
    """Group per-CCD CIF files ({CCD}.cif) into per-letter components.{X}.cif files."""
    os.makedirs(lib_dir, exist_ok=True)
    files = {}

    for cif_path in sorted(glob.glob(os.path.join(components_dir, "*.cif"))):
        first_char = os.path.basename(cif_path)[0]

        if first_char not in files:
            outfile_path = os.path.join(lib_dir, f"components.{first_char}.cif")
            files[first_char] = open(outfile_path, "w")

        with open(cif_path) as f:
            files[first_char].write(f.read().rstrip("\n") + "\n")

    for f in files.values():
        f.close()


def count_data_blocks(lib_dir: str) -> int:
    total_count = 0

    for filename in glob.glob(os.path.join(lib_dir, "components.*.cif")):
        with open(filename) as f:
            for line in f:
                if line.startswith("data_"):
                    total_count += 1

    return total_count


def main():
    parser = argparse.ArgumentParser(
        description="Build per-letter components.X.cif files for PyRosetta from ccd_cif/ and count data_ blocks."
    )

    parser.add_argument(
        "--lib-dir",
        required=True,
        help="Output directory for components.X.cif files",
    )

    parser.add_argument(
        "--components-dir",
        dest="components_dir",
        default=COMPONENTS_DIR,
        help="Directory of per-CCD CIF files (ccd_cif/)",
    )

    args = parser.parse_args()

    split_components(args.components_dir, args.lib_dir)

    total_count = count_data_blocks(args.lib_dir)
    print("Total number of data_ blocks:", total_count)


if __name__ == "__main__":
    main()
