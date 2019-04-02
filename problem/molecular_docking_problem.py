from math import pi, sin, cos
from problem.molecular_docking_energy import rosetta_energy_function

import numpy as np
import copy
import yaml

from problem.generic_problem import Problem


class MolecularDockingProblem(Problem):

    def __init__(self):
        super().__init__()
        print("Molecular Docking Problem")

        self.ATOM_TAG = ["HETATM", "ATOM"]
        self.BRANCH_TAG = "BRANCH"
        self.ENDBRANCH_TAG = "ENDBRANCH"
        self.END_TAG = "TORSDOF"

        self.pdb_pattern = "{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {" \
                           ":5s}{:2s}{:2s} "

        self.box_bounds = np.empty(3)
        self.center_bounds = np.empty(3)

        self.vina_path = ""
        self.vina_config = ""
        self.docking_complex = ""

        self.read_parameters()
        self.read_vina_config_file()

        self.mod_pos_atoms = []
        self.content = []
        self.pos_atoms = []
        self.branchs = {}
        self.index_branch = {}
        self.start_branchs = {}
        self.end_branchs = {}
        self.index_translate = {}
        self.index_atoms = {}
        self.index_ref = 0
        self.index_sec = 0

        self.atom_sec = None
        self.atom_ref = None
        self.partners = None

        self.get_bounds()
        self.is_multi_objective = False

        self.energy_function = rosetta_energy_function.RosettaScoringFunction()

    def read_parameters(self):
        with open("docking_config.yaml", 'r') as stream:
            try:
                config = yaml.load(stream)
                self.vina_path = config['vina_path']
                self.vina_config = config['vina_config']
                self.docking_complex = config['complex']

            except yaml.YAMLError as exc:
                print(exc)

    def read_vina_config_file(self):
        file = open("Docking/" + self.docking_complex + "/" + self.vina_config)

        while True:
            buffer_line = file.readline().split()

            if not buffer_line:
                break

            if buffer_line[0] == 'size_x':
                self.box_bounds[0] = float(buffer_line[2])
            elif buffer_line[0] == 'size_y':
                self.box_bounds[1] = float(buffer_line[2])
            elif buffer_line[0] == 'size_z':
                self.box_bounds[2] = float(buffer_line[2])

            elif buffer_line[0] == 'center_x':
                self.center_bounds[0] = float(buffer_line[2])
            elif buffer_line[0] == 'center_y':
                self.center_bounds[1] = float(buffer_line[2])
            elif buffer_line[0] == 'center_z':
                self.center_bounds[2] = float(buffer_line[2])

            elif buffer_line[0] == '#partners':
                self.partners = buffer_line[2]
            elif buffer_line[0] == '#ref':
                self.atom_ref = buffer_line[2]
            elif buffer_line[0] == '#sec':
                self.atom_sec = buffer_line[2]

    def get_bounds(self):
        self.lb = np.zeros(self.dimensions)
        self.ub = np.zeros(self.dimensions)

        for i in range(0, 3):
            self.lb[i] = float(self.box_bounds[i] / 2) * (-1)
            self.ub[i] = float(self.box_bounds[i] / 2)

        for i in range(3, 11):
            self.lb[i] = -pi
            self.ub[i] = pi

    def evaluate(self, angles):
        energy = 0.0

        self.perform_docking(angles)
        self.energy_function.update_ligand(self.get_dic_modified_atoms())
        energy = self.energy_function.evaluate_complex()

        if self.is_multi_objective:
            energy = self.evaluate_mo(angles)

        return energy

    def evaluate_mo(self, angles):
        intra_energy = 0.0
        inter_energy = 0.0

        return intra_energy, inter_energy

    def perform_docking(self, angles):
        self.mod_pos_atoms = np.array(copy.copy(self.pos_atoms))
        self.rotate_dihedral_angles(angles[4:])
        self.rotate_matrix(angles[3])
        self.translate_matrix([angles[0], angles[1], angles[2]])

    def rotate_dihedral_angles(self, theta):
        phi = 0
        psi = 0
        omega = 0
        for key in self.start_branchs.keys():
            vec_ref = np.array(
                [list(map(float, self.mod_pos_atoms[self.translate_position(self.branchs[key][0]) - 1]))[0] -
                 list(map(float, self.mod_pos_atoms[self.translate_position(self.branchs[key][1]) - 1]))[0],
                 list(map(float, self.mod_pos_atoms[self.translate_position(self.branchs[key][0]) - 1]))[1] -
                 list(map(float, self.mod_pos_atoms[self.translate_position(self.branchs[key][1]) - 1]))[1],
                 list(map(float, self.mod_pos_atoms[self.translate_position(self.branchs[key][0]) - 1]))[2] -
                 list(map(float, self.mod_pos_atoms[self.translate_position(self.branchs[key][1]) - 1]))[2]])

            angle = theta[key - 1]
            reference = self.mod_pos_atoms[self.translate_position(self.branchs[key][0]) - 1]

            for i in range(self.start_branchs[key], self.end_branchs[key] + 1):
                self.mod_pos_atoms[i] = self.translate_to_ref(self.mod_pos_atoms[i], reference, (0.0, 0.0, 0.0))[0]
                self.mod_pos_atoms[i] = self.rotate_matrix_theta(angle, vec_ref, [self.mod_pos_atoms[i]])[0]
                self.mod_pos_atoms[i] = self.translate_to_ref(self.mod_pos_atoms[i], (0.0, 0.0, 0.0), reference)[0]

    def translate_position(self, index):
        return self.index_translate.get(index)

    def translate_to_ref(self, matrix, reference, origin):
        translation = origin - reference
        translation = np.array([translation] * len(matrix))
        return matrix + translation

    def rotate_matrix(self, angle):
        # keep the position of reference atom
        old_origin = self.mod_pos_atoms[self.index_ref]
        # translate matrix in order to put the coordinates of ref atom in 0 origin
        self.mod_pos_atoms = self.translate_to_ref(self.mod_pos_atoms, self.mod_pos_atoms[self.index_ref],
                                                   (0.0, 0.0, 0.0))
        # build the ref vector
        vector_ref = [self.mod_pos_atoms[self.index_ref][0] - self.mod_pos_atoms[self.index_sec][0],
                      self.mod_pos_atoms[self.index_ref][1] - self.mod_pos_atoms[self.index_sec][1],
                      self.mod_pos_atoms[self.index_ref][2] - self.mod_pos_atoms[self.index_sec][2]]

        self.mod_pos_atoms = self.rotate_matrix_theta(angle, vector_ref, self.mod_pos_atoms)
        self.mod_pos_atoms = self.translate_to_ref(self.mod_pos_atoms, self.mod_pos_atoms[self.index_ref], old_origin)

    def rotate_matrix_theta(self, theta, reference_vector, matrix):
        vector = copy.copy(reference_vector)
        vector = self.normalize(vector)
        cos_value = cos(theta)
        sin_value = sin(theta)
        t_value = 1.0 - cos_value

        x = vector[0]
        y = vector[1]
        z = vector[2]

        rotationMatrix = np.array(
            [[(t_value * x * x) + cos_value, (t_value * x * y) - (sin_value * z), (t_value * x * z) + (sin_value * y)],
             [(t_value * x * y) + (sin_value * z), (t_value * y * y) + cos_value, (t_value * y * z - (sin_value * x))],
             [(t_value * x * z) - (sin_value * y), (t_value * y * z) + (sin_value * x), (t_value * z * z) + cos_value]])

        for i in range(len(matrix)):
            result = self.multiply_matrix(rotationMatrix, [[matrix[i][0]], [matrix[i][1]], [matrix[i][2]]])
            matrix[i][0] = result[0][0]
            matrix[i][1] = result[1][0]
            matrix[i][2] = result[2][0]

        return matrix

    def multiply_matrix(self, matrix_A, matrix_B):
        lines_A = len(matrix_A)
        columns_A = len(matrix_A[0])
        lines_B = len(matrix_B)
        columns_B = len(matrix_B[0])

        if columns_A == lines_B:
            dimension = columns_A
            matrix_result = [[sum(matrix_A[m][n] * matrix_B[n][p] for n in range(dimension))
                              for p in range(columns_B)] for m in range(lines_A)]
            return matrix_result

        else:
            return -1

    def get_dic_modified_atoms(self):
        modified_atoms = {}

        for key in self.index_atoms:
            modified_atoms[key] = self.mod_pos_atoms[self.index_atoms[key]]

        return modified_atoms

    # translate matrix by x, y, z values
    def translate_matrix(self, translation):
        translation = np.array([translation] * len(self.pos_atoms))
        self.mod_pos_atoms = self.mod_pos_atoms + translation

    def normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector

        return vector / norm

    def rosetta_energy_function_config(self):
        self.energy_function.partners = self.partners