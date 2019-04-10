# TODO list the objects used, avoiding the import of the whole lib

import pyrosetta
import sys


# TODO Refactor the whole class due to architectural issues.
class RosettaScoringFunction:

    def __init__(self, complex):
        pyrosetta.init(extra_options='-mute all')

        self.ligand_params = ""
        self.pdb_file_name = "instances/" + complex + "/" + complex + ".pdb"
        self.partners = ""
        self.pose = None
        self.res_set = None
        self.dock_jump = 1
        self.ligand_residue = None
        self.pos_atoms = {}
        self.modified_atoms = None

        self.single_scorefxn = pyrosetta.get_fa_scorefxn()
        self.scorefxn = pyrosetta.create_score_function("ligand")

    def set_ligand_params(self, parameter_file):
        if len(parameter_file) != 0 and parameter_file[0] != '':
            self.ligand_params = pyrosetta.Vector1(parameter_file.split(','))
        return None

    def load_pose(self):
        self.pose = pyrosetta.Pose()
        self.res_set = self.pose.conformation().modifiable_residue_type_set_for_conf()
        self.res_set.read_files_for_base_residue_types(self.ligand_params)
        self.pose.conformation().reset_residue_type_set_for_conf(self.res_set)
        pyrosetta.rosetta.core.import_pose.pose_from_file(self.pose, self.pdb_file_name)

        pyrosetta.rosetta.protocols.docking.setup_foldtree(self.pose, self.partners, pyrosetta.Vector1([self.dock_jump]))
        self.ligand_residue = self.pose.residue(self.pose.total_residue())

        self.load_coordinates()
        self.update_ligand(None)

    def load_coordinates(self):
        self.pos_atoms = {}

        for atom in range(self.ligand_residue.natoms()):
            coord = self.ligand_residue.xyz(atom + 1)
            atom_name = self.ligand_residue.atom_name(atom+1).split()[0]#.encode('ascii', 'ignore')
            self.pos_atoms[atom_name] = coord

    def update_ligand(self, dic):
        self.pose.energies().clear()

        if dic is not None:
            for key in self.pos_atoms:
                vec = pyrosetta.rosetta.numeric.xyzVector_double_t(dic[key][0], dic[key][1], dic[key][2])
                self.pose.residue(self.pose.total_residue()).set_xyz(self.ligand_residue.atom_index(key), vec)

    def evaluate_complex(self):
        return self.scorefxn(self.pose)
