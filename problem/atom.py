class Atom(object):
    """Read the PDBQT's content and split values in variables to better access"""
    tag = ''
    serial = ''
    atom = ''
    locIndicator = ''
    residue = ''
    chainID = ''
    seqResidue = ''
    insResidue = ''
    xCor = ''
    yCor = ''
    zCor = ''
    occupancy = ''
    temperature = ''
    segmentID = ''
    symbol = ''
    chargeAtom = ''
    line = ''
    bFrom = ''
    bTo = ''
    ATOM_TAG = ["HETATM", "ATOM"]
    BRANCH_TAG = "BRANCH"
    ENDBRANCH_TAG = "ENDBRANCH"

    def __init__(self, line):
        if line[0:6].strip() in self.ATOM_TAG:
            self.tag = line[0:6]
            self.serial = line[6:11]
            self.atom = line[12:16]
            self.locIndicator = line[16:17]
            self.residue = line[17:20]
            self.chainID = line[21:22]
            self.seqResidue = line[22:26]
            self.insResidue = line[26:27]
            self.xCor = line[30:38]
            self.yCor = line[38:46]
            self.zCor = line[46:54]
            self.occupancy = line[54:60]
            self.temperature = line[60:66]
            self.segmentID = line[70:76]
            self.symbol = line[76:78]
            self.chargeAtom = line[78:-1]

        elif line[0:6] == self.BRANCH_TAG:
            self.tag = line[0:6]
            self.bFrom = line[6:11]
            self.bTo = line[11:15]

        elif line[0:9] == self.ENDBRANCH_TAG:
            self.tag = line[0:9]
            self.bFrom = line[9:13]
            self.bTo = line[13:17]

        self.line = line

    def get_tag(self):
        return self.tag.strip()

    def get_serial(self):
        return int(self.serial.strip())

    def get_atom(self):
        return self.atom.strip()

    def get_residue(self):
        return self.residue.strip()

    def get_pos(self):
        return map(float, (self.xCor, self.yCor, self.zCor))

    def get_content(self):
        return self.line
