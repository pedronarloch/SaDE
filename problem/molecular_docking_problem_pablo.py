import math
from math import *

from aplIndividualGenerator import *
from atom import *
from problem import *
from randomIndividualGenerator import *
from rosettaFunction import *
from vinaFunction import *

from individual import *


class Docking(Problem):
    """Docking class"""
    # http://cupnet.net/pdb-format/
    pdbPattern = "{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:5s}{:2s}{:2s}"
    indexRef = 0
    indexSec = 0
    branchs = {}
    indexBranch = {}
    startBranchs = {}
    endBranchs = {}
    numBranchs = 0
    outputPath = ''
    ATOM_TAG = ["HETATM", "ATOM"]
    BRANCH_TAG = "BRANCH"
    ENDBRANCH_TAG = "ENDBRANCH"
    END_TAG = "TORSDOF"
    indexAtoms = {}
    originalPosAtoms = []
    posAtoms = []
    modPosAtoms = []
    content = []
    atomRef = None
    atomSec = None
    indexTranslate = {}
    scoringFunction = None
    function = None
    VINA = "vina"
    ROSETTA = "rosetta"
    cubeSize = [11, 11, 11]
    initialState = None
    loadInitialState = False

    def __init__(self):
        super(Docking, self).__init__()
        self.name = "docking"
        self.docking = True
        self.indexAtoms = {}
        self.posAtoms = []
        self.content = []
        self.branchs = {}
        self.indexBranch = {}
        self.startBranchs = {}
        self.endBranchs = {}
        self.indexTranslate = {}

        # Get the parameters
        self.instancePath = Utils.get_instance_path()
        self.instanceChangedPath = Utils.get_instance_changed_path()
        self.outputPath = Utils.get_output_path()
        self.scoringFunction = Utils.get_scoring_function()

        if self.scoringFunction == self.VINA:
            self.function = VinaFunction()

        elif self.scoringFunction == self.ROSETTA:
            self.function = RosettaFunction()

        self.atomRef = Utils.atomRef
        self.atomSec = Utils.atomSec
        self.loadInitialState = Utils.loadInitialConfig

        self.read_ligand_file()
        self.originalPosAtoms = copy.copy(self.posAtoms)
        self.load_search_range()
        self.randomize_ligand()

    def set_output_path(self, path):
        self.outputPath = path

    # read config file
    def read_config_file(self):
        file = open(Utils.get_config_path(), "r")
        var = True

        while var:
            bufferLine = file.readline().split()

            if len(bufferLine) > 0:
                if (bufferLine[0] == "size_x"):
                    self.cubeSize.append(float(bufferLine[2]))

                elif (bufferLine[0] == "size_y"):
                    self.cubeSize.append(float(bufferLine[2]))

                elif (bufferLine[0] == "size_z"):
                    self.cubeSize.append(float(bufferLine[2]))

            else:
                var = False

        file.close()

    # read original file of ligand to get base informations
    def read_ligand_file(self):
        finish = False
        file = open(self.instancePath, "r")
        countStartBranchs = 1

        while not finish:
            line = file.readline()

            if not line:
                finish = True

            else:
                atom = Atom(line)
                self.content.append(atom)

                if atom.get_tag() == self.END_TAG:
                    finish = True

                else:
                    if (atom.get_tag() in self.ATOM_TAG):
                        atomName = atom.get_atom()
                        self.indexAtoms[atomName] = len(self.posAtoms)
                        self.posAtoms.append(atom.get_pos())

                        # get central atom's index (reference to bigger cube)
                        if atomName == self.atomRef:
                            self.indexRef = len(self.posAtoms) - 1

                        elif atomName == self.atomSec:
                            self.indexSec = len(self.posAtoms) - 1

                        self.indexTranslate[atom.get_serial()] = len(self.posAtoms)

                    # get ligand branchs (ligand's rotation points)
                    elif atom.get_tag() == self.BRANCH_TAG:
                        branch = line.split()
                        self.branchs[countStartBranchs] = (int(branch[1]), int(branch[2]))
                        self.startBranchs[countStartBranchs] = len(self.posAtoms)
                        self.indexBranch[branch[1] + branch[2]] = countStartBranchs
                        countStartBranchs += 1

                    elif atom.get_tag() == self.ENDBRANCH_TAG:
                        branch = line.split()
                        self.endBranchs[self.indexBranch[branch[1] + branch[2]]] = len(self.posAtoms) - 1

        file.close()
        self.numBranchs = len(self.indexBranch)
        self.dimension = 4 + self.numBranchs

    def load_search_range(self):
        parameters = Utils.get_problem_parameters()
        vec = parameters.get("range").split(",")

        i = 0
        while i < len(vec):
            if vec[i] == 'X':
                tam = self.numBranchs + 1
            else:
                tam = int(vec[i])

            for j in range(tam):
                self.searchRange.append([self.translate_value(vec[i + 1]), self.translate_value(vec[i + 2])])
            i += 3

    def load_initial_ligand_state(self):
        individual = Individual(self.dimension)

        with open(Utils.populationPath) as file:
            line = file.readline()

        for value in line.split():
            individual.representation.append(float(value))

        return individual

    def randomize_ligand(self):
        algorithmParameters = Utils.get_algorithm_parameters()
        iGenerator = algorithmParameters.get("individualGenerator")

        if (iGenerator == "random"):
            indGenerator = RandomIndividualGenerator(True)

        else:
            indGenerator = APLIndividualGenerator()
            indGenerator.load_apl(Utils.aplVector)

        if self.loadInitialState:
            individual = self.load_initial_ligand_state()
        else:
            indGenerator.set_dimension(self.dimension)
            indGenerator.set_search_range(self.searchRange)
            individual = indGenerator.generate_individual()

        self.initialState = individual
        self.perform_docking(individual.representation)
        self.write_ligand(self.instanceChangedPath)
        self.posAtoms = copy.copy(self.modPosAtoms)

        # update search range in problem
        self.searchRange[0] = [self.searchRange[0][0] - individual.representation[0],
                               self.searchRange[0][1] - individual.representation[0]]
        self.searchRange[1] = [self.searchRange[1][0] - individual.representation[1],
                               self.searchRange[1][1] - individual.representation[1]]
        self.searchRange[2] = [self.searchRange[2][0] - individual.representation[2],
                               self.searchRange[2][1] - individual.representation[2]]

    # print "range updated", self.searchRange

    # self.modPosAtoms = np.array( copy.copy( self.posAtoms ) )
    # self.write_ligand( "teste.pdb", "pdb" )

    def translate_value(self, value):
        if value == "pi" or value == "math.pi":
            return pi

        elif value == "-pi" or value == "-math.pi":
            return -pi

        elif value == "-cubeSize":
            return -self.cubeSize[0] / 2.0

        elif value == "cubeSize":
            return self.cubeSize[0] / 2.0

        else:
            return float(value)

    # get the real position of an atom
    def translate_position(self, index):
        return self.indexTranslate.get(index)

    def calculate_points_distance(self, pointA, pointB):
        return math.sqrt(
            math.pow(pointA[0] - pointB[0], 2) + math.pow(pointA[1] - pointB[1], 2) + math.pow(pointA[2] - pointB[2],
                                                                                               2))

    # multiply two matrix
    def multiply_matrix(self, matrixA, matrixB):
        linesA = len(matrixA)
        columnsA = len(matrixA[0])
        linesB = len(matrixB)
        columnsB = len(matrixB[0])

        if columnsA == linesB:
            dimension = columnsA
            matrixResult = [[sum(matrixA[m][n] * matrixB[n][p] for n in range(dimension)) \
                             for p in range(columnsB)] for m in range(linesA)]
            return matrixResult

        else:
            return -1

    # translate matrix with 2 origin points (old and new)
    def translate_to_ref(self, matrix, reference, origin):
        translation = origin - reference
        translation = np.array([translation] * len(matrix))
        return matrix + translation

    # normalize vector
    def normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector

        return vector / norm

    # rotate structure in function by angle
    def rotate_matrix_theta(self, theta, vectorReference, matrix):
        vector = copy.copy(vectorReference)
        vector = self.normalize(vector)
        cosValue = math.cos(theta)
        sinValue = math.sin(theta)
        tValue = 1.0 - cosValue
        x = vector[0]
        y = vector[1]
        z = vector[2]

        rotationMatrix = np.array(
            [[(tValue * x * x) + cosValue, (tValue * x * y) - (sinValue * z), (tValue * x * z) + (sinValue * y)],
             [(tValue * x * y) + (sinValue * z), (tValue * y * y) + cosValue, (tValue * y * z - (sinValue * x))],
             [(tValue * x * z) - (sinValue * y), (tValue * y * z) + (sinValue * x), (tValue * z * z) + cosValue]])

        for i in range(len(matrix)):
            result = self.multiply_matrix(rotationMatrix, [[matrix[i][0]], [matrix[i][1]], [matrix[i][2]]])
            matrix[i][0] = result[0][0]
            matrix[i][1] = result[1][0]
            matrix[i][2] = result[2][0]

        return matrix

    # rotate atoms in function of dihedral angles
    def rotate_dihedral_angles(self, theta):
        for key in self.startBranchs.keys():
            vecRef = np.array([self.modPosAtoms[self.translate_position(self.branchs[key][0]) - 1][0] -
                               self.modPosAtoms[self.translate_position(self.branchs[key][1]) - 1][0],
                               self.modPosAtoms[self.translate_position(self.branchs[key][0]) - 1][1] -
                               self.modPosAtoms[self.translate_position(self.branchs[key][1]) - 1][1],
                               self.modPosAtoms[self.translate_position(self.branchs[key][0]) - 1][2] -
                               self.modPosAtoms[self.translate_position(self.branchs[key][1]) - 1][2]])

            angle = theta[key - 1]
            reference = self.modPosAtoms[self.translate_position(self.branchs[key][0]) - 1]
            # print self.branchs[key][0], self.branchs[key][1]
            # print self.startBranchs[key], self.endBranchs[key] + 1
            # print "$", self.content[self.branchs[key][0]].atom, self.content[self.branchs[key][1]].atom
            for i in range(self.startBranchs[key], self.endBranchs[key] + 1):
                # print i
                self.modPosAtoms[i] = self.translate_to_ref(self.modPosAtoms[i], reference, (0.0, 0.0, 0.0))[0]
                self.modPosAtoms[i] = self.rotate_matrix_theta(angle, vecRef, [self.modPosAtoms[i]])[0]
                self.modPosAtoms[i] = self.translate_to_ref(self.modPosAtoms[i], (0.0, 0.0, 0.0), reference)[0]

    # rotate matrix by theta angle
    def rotate_matrix(self, angle):
        # keep the position of reference atom
        oldOrigin = self.modPosAtoms[self.indexRef]
        # translate matrix in order to put the coordinates of ref atom in 0 origin
        self.modPosAtoms = self.translate_to_ref(self.modPosAtoms, self.modPosAtoms[self.indexRef], (0.0, 0.0, 0.0))
        # build the ref vector
        vectorRef = [self.modPosAtoms[self.indexRef][0] - self.modPosAtoms[self.indexSec][0],
                     self.modPosAtoms[self.indexRef][1] - self.modPosAtoms[self.indexSec][1],
                     self.modPosAtoms[self.indexRef][2] - self.modPosAtoms[self.indexSec][2]]

        # print self.indexRef, self.indexSec
        # print vectorRef
        self.modPosAtoms = self.rotate_matrix_theta(angle, vectorRef, self.modPosAtoms)
        self.modPosAtoms = self.translate_to_ref(self.modPosAtoms, self.modPosAtoms[self.indexRef], oldOrigin)

    # translate matrix by x, y, z values
    def translate_matrix(self, translation):
        translation = np.array([translation] * len(self.posAtoms))
        self.modPosAtoms = self.modPosAtoms + translation

    # perform docking of ligand based in translation and rotation of its structure
    # and in the internal rotation of dihedral angles
    def perform_docking(self, individual, path=''):
        self.modPosAtoms = np.array(copy.copy(self.posAtoms))
        # rotate dihedral angles
        self.rotate_dihedral_angles(individual[4:])
        # rotate structure
        self.rotate_matrix(individual[3])
        # translate structure
        self.translate_matrix([individual[0], individual[1], individual[2]])

        if (self.scoringFunction == "vina"):
            if path == '':
                path = self.outputPath

            # write PDB with new atom positions
            self.write_ligand(path)

    def get_dic_modified_atoms(self):
        modifiedAtoms = {}

        for key in self.indexAtoms:
            modifiedAtoms[key] = self.modPosAtoms[self.indexAtoms[key]]

        return modifiedAtoms

    # perform docking and evaluate solution
    def evaluate_solution(self, solution):
        self.perform_docking(solution.get_representation())
        self.function.set_modified_atoms(self.get_dic_modified_atoms())
        solution.set_score(self.function.scores())

    def get_cube_size(self):
        return self.cubeSize

    def set_dimension(self, dimension):
        self.numBranchs = dimension - 4

    def get_function_number(self):
        return Utils.get_instance()

    # write content of ligand
    def write_ligand(self, path, fileFormat="pdbqt"):
        pdbNew = open(path, "w")
        countTotal = 1

        for key in range(0, len(self.content)):
            if (self.content[key].get_tag() in self.ATOM_TAG):
                pdbNew.write(self.pdbPattern.format(self.content[key].get_tag(), self.content[key].get_serial(),
                                                    self.content[key].get_atom(), self.content[key].locIndicator,
                                                    self.content[key].residue, self.content[key].chainID,
                                                    int(self.content[key].seqResidue), self.content[key].insResidue,
                                                    float(self.modPosAtoms[countTotal - 1][0]),
                                                    float(self.modPosAtoms[countTotal - 1][1]),
                                                    float(self.modPosAtoms[countTotal - 1][2]),
                                                    float(self.content[key].occupancy),
                                                    float(self.content[key].temperature), self.content[key].segmentID,
                                                    self.content[key].symbol, self.content[key].chargeAtom) + "\n")
                countTotal += 1
            else:
                if fileFormat == "pdbqt":
                    pdbNew.write(str(self.content[key].get_content()))

        pdbNew.close()
