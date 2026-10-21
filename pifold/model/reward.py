import pyrosetta
import os
import logging
import pickle
import subprocess

from copy import deepcopy
from typing import Any, Optional
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    TimeoutError,
)

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from configs.configs_data import data_configs

RKDIT_MOL_PKL = data_configs["ccd_components_rdkit_mol_file"]
MOL_ENERGY_PKL = data_configs["ccd_components_mol_energy_file"]


def worker_init(extra_init_args: list[str]) -> None:
    """
    Initializer for each worker process. This is where PyRosetta is initialized
    silently. This is called only ONCE when each worker process starts.
    """
    init_args = [
        "-constant_seed true",
        "-out:level 0",
        "-obey_ssbond true",
        "-ignore_zero_occupancy false",
        "-ignore_unrecognized_res true",
    ]

    pyrosetta_logger = logging.getLogger("pyrosetta")
    pyrosetta_logger.setLevel(logging.ERROR)

    if extra_init_args:
        init_args.extend(extra_init_args)

    pyrosetta.init(" ".join(init_args), silent=True)


def score_one_file(
    pdb_path: str,
    scorefxn_name: str,
    extra_terms: list[str] = None,
) -> dict:
    """
    Scores a single PDB file, creating a fresh score function for each task.
    """
    if not os.path.exists(pdb_path):
        return {"error": "File not found", "success": False}

    try:
        scorefxn = pyrosetta.create_score_function(scorefxn_name)
        if extra_terms:
            for term, weight in extra_terms.items():
                try:
                    score_type = getattr(pyrosetta.rosetta.core.scoring, term)
                    scorefxn.set_weight(score_type, weight)
                except AttributeError:
                    print(
                        f"Warning: Score term '{term}' not found and will be skipped."
                    )

        pose = pyrosetta.pose_from_file(pdb_path)
        total_score = scorefxn(pose)
        num_residues = pose.total_residue()

        energy_terms = {}
        energies = pose.energies()
        score_types = scorefxn.get_nonzero_weighted_scoretypes()

        for score_type in score_types:
            term_name = str(score_type).split(".")[1]
            raw_score = energies.total_energies()[score_type]
            energy_terms[term_name] = round(raw_score, 3)

        return {
            "score": round(total_score, 3),
            "norm_score": round(total_score / num_residues, 3),
            "energy_terms": energy_terms,
            "success": True,
        }
    except Exception as e:
        return {"score": None, "energy_terms": {}, "success": False, "error": str(e)}


class PyrosettaScorer:
    """
    A silent, parallel reward model that uses a pool of workers to score
    protein structures. It correctly handles process initialization and task isolation.
    """

    def __init__(
        self,
        scorefxn_name: str = "beta_nov16",
        num_workers: int = 10,
        use_tqdm: bool = True,
        timeout: int = 20,
        max_score: float = 25,
        min_score: float = -3,
        extra_init_args: list = None,
        extra_terms: dict = None,
    ):
        self.scorefxn_name = scorefxn_name
        self.extra_init_args = extra_init_args if extra_init_args is not None else []
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.timeout = timeout
        self.use_tqdm = use_tqdm
        self.max_score = max_score
        self.min_score = min_score
        self.extra_terms = extra_terms if extra_terms is not None else {}
        if self.scorefxn_name in ["beta_nov16", "beta_jan25"]:
            beta_flag = f"-corrections:{self.scorefxn_name}"
            if beta_flag not in self.extra_init_args:
                self.extra_init_args.append(beta_flag)

    def score(self, file_list: list) -> tuple[dict, torch.Tensor]:
        """
        Scores a list of structure files (CIF, PDB) in parallel.
        """
        results = {}
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=worker_init,
            initargs=(self.extra_init_args,),
        ) as executor:
            futures = {
                executor.submit(
                    score_one_file, path, self.scorefxn_name, self.extra_terms
                ): path
                for path in file_list
            }

            if self.use_tqdm:
                progress = tqdm(
                    as_completed(futures), total=len(file_list), desc="Scoring Files"
                )
            else:
                progress = as_completed(futures)

            for future in progress:
                filepath = futures[future]
                try:
                    results[filepath] = future.result(timeout=self.timeout)
                except TimeoutError:
                    results[filepath] = {
                        "success": False,
                        "error": f"Task timed out after {self.timeout} seconds",
                    }
                except Exception as e:
                    results[filepath] = {"success": False, "error": str(e)}

        scores = []
        for path in file_list:
            result = results.get(path)
            if result and result.get("success"):
                scores.append(result["norm_score"])
            else:
                scores.append(float("nan"))  # Use NaN for failed scores

        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        scores_tensor = torch.clamp(scores_tensor, self.min_score, self.max_score)
        scores_tensor = 1 - (scores_tensor - self.min_score) / (
            self.max_score - self.min_score
        )
        return results, scores_tensor


def score_one_ligand(
    protein_path: str, ligand_path: str, executable_path: str, scorefxn_name: str
) -> dict:
    """Scores a single protein-ligand pair using a command-line executable."""
    if not os.path.exists(protein_path) or not os.path.exists(ligand_path):
        return {"success": False, "error": "File not found"}
    command = [
        executable_path,
        "-r",
        protein_path,
        "-l",
        ligand_path,
        "--scoring",
        scorefxn_name,
        "--score_only",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=10
        )
        score = next(
            (
                float(line.split()[1])
                for line in result.stdout.splitlines()
                if line.strip().startswith("Affinity:")
            ),
            None,
        )
        if score is None:
            return {"success": False, "error": "Could not parse score"}
        with open(ligand_path) as f:
            num_atoms = int(
                [line for line in f if line.strip().endswith("V2000")][0].split()[0]
            )
        return {"score": score, "norm_score": score / num_atoms, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


class SminaScorer:
    """A scorer for protein-ligand complexes using Smina."""

    def __init__(
        self,
        executable_path: str,
        scorefxn_name: str,
        use_tqdm: bool = True,
        num_workers: int = 10,
        max_score: float = 0,
        min_score: float = -13,
        timeout: int = 10,
    ):

        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"Smina executable not found at: {executable_path}")
        self.executable = executable_path
        self.scorefxn_name = scorefxn_name
        self.num_workers = num_workers
        self.use_tqdm = use_tqdm
        self.max_score = max_score
        self.min_score = min_score
        self.timeout = timeout

    def score(self, protein_ligand_dict: dict) -> tuple[dict, torch.Tensor]:
        """
        Scores a dictionary of protein-ligand pairs in parallel.
        """
        results = {}
        tasks = [(p, l[0]) for p, l_list in protein_ligand_dict.items() for l in l_list]

        if not tasks:
            return None, torch.zeros((len(protein_ligand_dict)), dtype=torch.float32)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    score_one_ligand,
                    protein,
                    ligand,
                    self.executable,
                    self.scorefxn_name,
                ): (protein, ligand)
                for protein, ligand in tasks
            }

            if self.use_tqdm:
                progress = tqdm(
                    as_completed(futures), total=len(tasks), desc="Smina Scoring"
                )
            else:
                progress = as_completed(futures)

            for future in progress:
                pair = futures[future]
                try:
                    results[pair] = future.result(timeout=self.timeout)
                except TimeoutError:
                    results[pair] = {
                        "success": False,
                        "error": f"Task timed out after {self.timeout} seconds",
                    }
                except Exception as e:
                    results[pair] = {"success": False, "error": str(e)}

        scores_dict = {}
        for protein, ligand_list in protein_ligand_dict.items():
            protein_scores = []
            for ligand, _ in ligand_list:
                result = results.get((protein, ligand))
                if result and result.get("success"):
                    protein_scores.append(result["score"])
                else:
                    protein_scores.append(float("nan"))  # Use NaN for failed scores

            scores_dict[protein] = torch.tensor(protein_scores, dtype=torch.float32)

        scores_tensor = torch.stack([value for value in scores_dict.values()], dim=0)
        scores_tensor = scores_tensor.mean(dim=1, keepdim=False)
        scores_tensor = torch.clamp(scores_tensor, self.min_score, self.max_score)
        scores_tensor = 1 - (scores_tensor - self.min_score) / (
            self.max_score - self.min_score
        )
        return results, scores_tensor


def calculate_energy(mol: Chem.Mol) -> float:
    """
    Returns: energy (NaN if failed)
    """
    mol = Chem.Mol(mol)  # Make a deep copy of the molecule
    mol = Chem.AddHs(mol, addCoords=True)

    try:
        ff = UFFGetMoleculeForceField(mol, confId=-1)
    except Exception:
        return np.nan
    try:
        energy = ff.CalcEnergy()
    except:
        return np.nan
    return energy


def relax_constrained(mol: Chem.Mol, maxDispl=0.1) -> float:
    """
    maxDispl: maximum displacement for position constraints during local relaxation.
    Returns:
        relaxed molecule: (returns None if fail)
    """
    mol = deepcopy(mol)  # Make a deep copy of the molecule
    mol = Chem.AddHs(mol, addCoords=True)
    try:
        ff = UFFGetMoleculeForceField(mol, confId=-1)
    except Exception:
        return np.nan

    for i in range(mol.GetNumAtoms()):
        ff.UFFAddPositionConstraint(i, maxDispl=maxDispl, forceConstant=1.0e5)
    try:
        ff.Minimize()
        return mol
    except:
        None


class StrainEngergyScorer:
    def __init__(
        self,
        num_workers: int = 4,
        timeout: int = 5,
        use_tqdm: bool = True,
        maxDispl: float = 0.1,
        max_score: float = 400,
        min_score: float = 0,
    ):
        """Rule-based bond reward model using RDKit reference molecules."""
        self.timeout = timeout
        self.num_workers = num_workers
        self.use_tqdm = use_tqdm
        self.maxDispl = maxDispl
        self.max_score = max_score
        self.min_score = min_score
        with open(MOL_ENERGY_PKL, "rb") as f:
            self.mols_energy = pickle.load(f)

    def extract_ccd_from_path(self, path: str) -> str:
        """Extract CCD code from ligand file path."""
        base = os.path.basename(path)
        parts = base.split("ligand_")[-1].split("_")
        return parts[0]

    def score_ligand(self, ligand_items: list[str, Chem.Mol]) -> dict:
        try:
            ligand_path, pred_mol = ligand_items
            ccd = self.extract_ccd_from_path(ligand_path)
            true_mol_energy = self.mols_energy[ccd]

            Chem.SanitizeMol(pred_mol)
            locally_relaxed_mol = relax_constrained(pred_mol, maxDispl=self.maxDispl)
            local_energy = calculate_energy(locally_relaxed_mol)
            strain_energy = local_energy - true_mol_energy

            return {
                "success": True,
                "ccd": ccd,
                "score": strain_energy,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "score": float("nan")}

    # --- Main scoring interface ---
    def score(self, protein_ligand_dict: dict) -> tuple[dict, torch.Tensor]:
        """
        Scores a dictionary of protein-ligand pairs in parallel.
        Returns (results_dict, scores_tensor).
        """
        results = {}
        tasks = [
            (protein, ligand)
            for protein, lig_list in protein_ligand_dict.items()
            for ligand in lig_list
        ]
        # Handle empty input
        if not tasks:
            empty_scores = torch.zeros((len(protein_ligand_dict)), dtype=torch.float32)
            return None, empty_scores

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.score_ligand, ligand_items): (
                    protein,
                    ligand_items,
                )
                for protein, ligand_items in tasks
            }

            if self.use_tqdm:
                progress = tqdm(
                    as_completed(futures),
                    total=len(tasks),
                    desc="Strain Energy Scoring",
                )
            else:
                progress = as_completed(futures)

            for future in progress:
                pair = futures[future]
                pair = (pair[0], pair[1][0])
                try:
                    results[pair] = future.result(timeout=self.timeout)
                except TimeoutError:
                    results[pair] = {
                        "success": False,
                        "error": f"Task timed out after {self.timeout} seconds",
                    }
                except Exception as e:
                    results[pair] = {"success": False, "error": str(e)}

        scores_dict = {}
        for protein, ligand_list in protein_ligand_dict.items():
            ligand_scores = []
            for ligand, _ in ligand_list:
                result = results.get((protein, ligand))
                if result and result.get("success"):
                    ligand_scores.append(result["score"])
                else:
                    # rdkit cannot read the mol due to bond error, assign max score
                    ligand_scores.append(self.max_score)
            scores_dict[protein] = torch.tensor(ligand_scores, dtype=torch.float32)

        scores_tensor = torch.stack([v for v in scores_dict.values()], dim=0)
        scores_tensor = scores_tensor.mean(dim=1, keepdim=False)
        scores_tensor = torch.clamp(scores_tensor, self.min_score, self.max_score)
        scores_tensor = 1 - (scores_tensor - self.min_score) / (
            self.max_score - self.min_score
        )
        return results, scores_tensor


class RewardModel:
    def __init__(
        self,
        pyrosetta_config: dict[str, Any],
        smina_config: dict[str, Any],
        strain_energy_config: dict[str, Any],
        clean_files: bool = True,
        ligand: bool = False,
    ):
        self.pyrosetta_scorer = PyrosettaScorer(**pyrosetta_config)
        self.smina_scorer = SminaScorer(**smina_config)
        self.strain_energy_scorer = StrainEngergyScorer(**strain_energy_config)
        self.clean_files = clean_files
        self.ligand = ligand

    def _cleanup(
        self, structure_files: list[str], ligand_files: Optional[dict] = None
    ) -> None:
        """Parallel cleanup taking structure and ligand inputs directly."""
        # Flatten all paths into a single list
        file_paths = set(structure_files)
        apo_file_paths = list(ligand_files.keys())
        file_paths.update(apo_file_paths)

        if ligand_files:
            for lig_list in ligand_files.values():
                for item in lig_list:
                    # Handle both path string or [path, mol] tuple
                    path = item[0] if isinstance(item, (list, tuple)) else item
                    if isinstance(path, str):
                        file_paths.add(path)

        def delete_file(path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Error deleting {path}: {e}")

        # Execute in parallel using the worker count from the scorer
        with ThreadPoolExecutor(
            max_workers=self.pyrosetta_scorer.num_workers
        ) as executor:
            executor.map(delete_file, list(file_paths))

    def score(
        self,
        structure_files: list[str],
        ligand_files: Optional[dict[str, list[str]]] = None,
    ) -> tuple[
        tuple[dict, torch.Tensor],
        Optional[tuple[dict, torch.Tensor]],
        Optional[tuple[dict, torch.Tensor]],
    ]:
        """
        Scores structures and, if provided, ligands.

        Args:
            structure_files (list): A list of protein file paths.
            ligand_files (dict, optional): Maps protein paths to lists of ligand paths.

        Returns:
            - If ligand_files is None: A single tensor of protein scores.
            - If ligand_files is provided: A tuple of (protein_scores_tensor, ligand_scores_dict).
        """
        apo_structure_files = (
            list(ligand_files.keys()) if ligand_files is not None else structure_files
        )
        protein_results = self.pyrosetta_scorer.score(apo_structure_files)

        ligand_results = (None, None)
        strain_results = (None, None)
        if self.ligand:
            ligand_results = self.smina_scorer.score(ligand_files)
            strain_results = self.strain_energy_scorer.score(ligand_files)

        if self.clean_files:
            self._cleanup(structure_files, ligand_files)

        return protein_results, ligand_results, strain_results
