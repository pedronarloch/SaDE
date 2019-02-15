import random

import angles
import numpy as np
import pyrosetta
import yaml
from pyrosetta.rosetta.core.scoring import CA_rmsd
from pyrosetta.rosetta.core.scoring.dssp import Dssp

from problems.angle_prob_list import hists as apl
from problems.generic_problem import Problem


class ProteinStructurePredictionProblem(Problem):

    def __init__(self):
        super().__init__()
        print("Protein Structure Prediction Problem Instantied!")
        self.read_parameters()
        self.start_rosetta_energy_function()
        self.get_bounds()
        self.apl_hist = apl.HistogramFiles(self.protein, self.fasta, self.secondary_structure, 3)

    def start_rosetta_energy_function(self):
        pyrosetta.init()

        if self.rosetta_energy_type == 'centroid':
            self.scorefxn = pyrosetta.create_score_function('score3')
            self.general_pose = pyrosetta.pose_from_sequence(self.fasta, "centroid")
            self.pose_2 = pyrosetta.pose_from_sequence(self.fasta, "centroid")

            # self.dimensions = len(self.fasta * 3)
            self.dimensions = len(self.fasta * 2)

        elif self.rosetta_energy_type == 'full_atom':
            self.scorefxn = pyrosetta.get_fa_scorefxn()
            self.general_pose = pyrosetta.pose_from_sequence(self.fasta, "fa_standard")

        else:
            print("There is not any energy type: ", self.rosetta_energy_type)

        # self.rcsb_protein = toolbox.pose_from_rcsb(self.protein)
        self.rcsb_protein = pyrosetta.pose_from_pdb(self.protein+".clean.pdb")

    def read_parameters(self):
        with open("psp_config.yaml", 'r') as stream:
            try:
                config = yaml.load(stream)
                self.protein = config['protein_name']
                self.fasta = config['fasta']
                self.secondary_structure = config['secondary_structure']
                self.rosetta_energy_type = config['rosetta_energy_type']
                self.use_ss_reinforcement = config['use_ss_reinforcement']
            except yaml.YAMLError as exc:
                print(exc)

    def evaluate(self, angles):
        energy = 0.0

        if self.rosetta_energy_type == 'centroid':
            energy = self.evaluate_by_centroid(angles)
        elif self.rosetta_energy_type == 'full_atom':
            energy = self.evaluate_by_full_atom(angles)

        return float(energy)

    def check_bounds_random(self, trial):
        for i in range(0, len(trial)):
            if trial[i] < -180:
                trial[i] = random.uniform(-180, 180)
            elif trial[i] > 180:
                trial[i] = random.uniform(-180, 180)

    def check_bounds(self, trial):
        for i in range(0, len(trial)):
            if trial[i] < -180:
                trial[i] = float(format(angles.normalize(trial[i], -180, 180), '.4f'))
            elif trial[i] > 180:
                trial[i] = float(format(angles.normalize(trial[i], -180, 180), '.4f'))

    def get_bounds(self):
        self.lb = np.zeros(self.dimensions)
        self.ub = np.zeros(self.dimensions)

        for i in range(0, self.dimensions):
            self.lb[i] = float(format(-180.00, '.4f'))
            self.ub[i] = float(format(180.00, '.4f'))

    def evaluate_by_centroid(self, angles):
        pose = self.create_pose_centroid(angles)
        score = float(format(self.scorefxn(pose), '.4f'))

        if self.use_ss_reinforcement:
            score += self.get_ss_reinforcement(pose)

        return score

    # TODO Include the Full Atom version
    def generate_apl_individual(self):
        generated_angles = self.apl_hist.read_histogram()
        aux = np.zeros(self.dimensions)
        index = 0
        for i in range(0, len(generated_angles)):
            aux[index] = generated_angles[i][0]
            aux[index + 1] = generated_angles[i][1]
            index += 2

        return aux

    def get_ss_reinforcement(self, pose):
        reinforcement = 0.0

        secondary_structure = Dssp(pose)
        secondary_structure.insert_ss_into_pose(pose)
        ss2 = pose.secstruct()

        for i in range(0, len(self.secondary_structure)):
            if self.secondary_structure[i] in ['H', 'G', 'I']:
                if ss2[i] == 'H':
                    reinforcement += -10.0
                else:
                    reinforcement += 10.0
            elif self.secondary_structure[i] in ['E', 'B', 'b']:
                if ss2[i] == 'E':
                    reinforcement += -10.0
                else:
                    reinforcement += 10.0
            elif self.secondary_structure[i] in ['C', 'T'] and ss2[i] != 'L':
                reinforcement += 10.0

        return reinforcement

    def create_pose_centroid(self, angles):
        # pose = pyrosetta.pose_from_sequence(self.fasta, "centroid")
        pose = self.general_pose
        index = 0
        for i in range(0, len(self.fasta)):
            pose.set_phi(i+1, angles[index])
            pose.set_psi(i+1, angles[index+1])
            # pose.set_omega(i+1, angles[index+2])
            pose.set_omega(i+1, 180)
            index += 2

        return pose

    def create_new_pose_centroid(self, angles):
        pose = self.pose_2  # pyrosetta.pose_from_sequence(self.fasta, "centroid")
        index = 0

        for i in range(0, len(self.fasta)):
            pose.set_phi(i+1, angles[index])
            pose.set_psi(i+1, angles[index+1])
            # pose.set_omega(i+1, angles[index+2])
            pose.set_omega(i+1, 180)
            index += 2

        return pose

    def generate_pdb(self, angles, name):
        pose = self.create_pose_centroid(angles)
        pose.dump_pdb(name)

        return pose

    def compare_rmsd_rcsb(self, pose):
        rmsd = CA_rmsd(self.rcsb_protein, pose)

        return rmsd

    def compare_rmsd(self, pose1, pose2):
        rmsd = CA_rmsd(pose1, pose2)

        return rmsd

    def compare_rmsd_v2(self, ind_1, ind_2):
        pose_1 = self.create_pose_centroid(ind_1)
        pose_2 = self.create_new_pose_centroid(ind_2)

        rmsd = self.compare_rmsd(pose_1, pose_2)

        return rmsd

    def get_num_side_chain_angles(self, amino_name):
        if (amino_name == 'GLY' or amino_name == 'G') or (amino_name == 'ALA' or amino_name == 'A') or (amino_name == 'PRO' or amino_name == 'P'):
            return 0
        elif (amino_name == 'SER' or amino_name == 'S') or (amino_name == 'CYS' or amino_name == 'C') or (amino_name == 'THR' or amino_name == 'T') or (amino_name == 'VAL' or amino_name == 'V'):
            return 1
        elif (amino_name == 'ILE' or amino_name == 'I') or (amino_name == 'LEU' or amino_name == 'L') or (amino_name == 'ASP' or amino_name == 'D') or (amino_name == 'ASN' or amino_name == 'N') or (amino_name == 'PHE' or amino_name == 'F') or (amino_name == 'TYR' or amino_name == 'Y') or (amino_name == 'HIS' or amino_name == 'H') or (amino_name == 'TRP' or amino_name == 'W'):
            return 2
        elif (amino_name == 'MET' or amino_name == 'M') or (amino_name == 'GLU' or amino_name == 'E') or (amino_name == 'GLN' or amino_name == 'Q'):
            return 3
        elif (amino_name == 'LYS' or amino_name == 'K') or (amino_name == 'ARG' or amino_name == 'R'):
            return 4
        return 0