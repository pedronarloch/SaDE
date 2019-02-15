import os
import random
import copy
import sys

class HistogramFiles:
	heat_maps = None
	APL = None
	protein = None
	primary_structure = None
	secondary_structure = None

	def __init__(self, protein, primary_structure, secondary_structure, heat_maps):
		self.protein = protein
		self.primary_structure = primary_structure
		self.secondary_structure = secondary_structure
		self.heat_maps = heat_maps
		self.carrega_apl()

	def carrega_apl(self):
		self.APL = {}
		path_apl = "./Proteins/"+self.protein+"/"
		print("Loading APL from: ", path_apl)

		if self.heat_maps == 1:
			list_apls = [self.protein+'-1']
		elif self.heat_maps == 3:
			list_apls = [self.protein+"-3", self.protein+"-2-Right", self.protein+"-2-Left", self.protein+"-1"]
		else:
			list_apls = [self.protein+'-1']

		dirs_apl = [x for x in next(os.walk(os.path.join(str(path_apl)+str(),'.')))[1] if not x.startswith(".")]

		for folder in dirs_apl:
			if folder.startswith(self.protein[:4]):
				if folder not in self.APL.keys():
					self.APL[folder] = {}

					if folder not in list_apls:
						continue

					files_apl = os.listdir(path_apl + folder)
					for file_apl in files_apl:
						id_apl = '_'.join(file_apl.split("_")[:-1])

						if (file_apl[-4:] == '.dat') and (file_apl[-4:] not in self.APL[folder].keys()):
							self.APL[folder][id_apl] = []

							with open(path_apl + folder + "/" + file_apl, 'r') as file:
								for line in file:
									list_line = line.split()
									self.APL[folder][id_apl].append([float(list_line[0]), float(list_line[1]), float(list_line[2]), eval(''.join(list_line[3:]))])
							self.APL[folder][id_apl].sort(key=lambda x: x[2], reverse=True)

							if len(self.APL[folder][id_apl]) == 0:
								remove = self.APL[folder].pop(id_apl)

	def read_histogram(self):
		aa_angles = []

		for i in range(len(self.primary_structure)):
			if i != 0 and i != len(self.primary_structure) - 1 and self.heat_maps == 3:
				aa_prv = self.primary_structure[i - 1]
				aa_cur = self.primary_structure[i]
				aa_nxt = self.primary_structure[i + 1]

				ss_prv = self.secondary_structure[i - 1]
				ss_cur = self.secondary_structure[i]
				ss_nxt = self.secondary_structure[i + 1]

				chance = random.random()

				prob = []

				if 0.0 <= chance < 0.5: #Have 50% to get from APL3 if it exists. If it doesn't exist, so search for APL2 and APL1

					key = aa_prv + aa_cur + aa_nxt + "_" + ss_prv + ss_cur + ss_nxt
					if key in self.APL[self.protein+'-3']:
						prob = self.APL[self.protein+'-3'][key]
					else:
						#50% to get righ/left APL-2
						if int(random.random()):
							key = aa_cur + aa_nxt + "_" + ss_cur + ss_nxt
							if key in self.APL[self.protein+'-2-Right']:
								prob = self.APL[self.protein+'-2-Right'][key]
							else:
								key = aa_cur + '_' + ss_cur
								prob = self.APL[self.protein+'-1'][key]
						else:
							key = aa_prv + aa_cur + '_' + ss_prv + ss_cur
							if key in self.APL[self.protein+'-2-Left']:
								prob = self.APL[self.protein+'-2-Left'][key]
							else:
								key = aa_cur + '_' + ss_cur
								prob = self.APL[self.protein+'-1'][key]

				elif 0.5 <= chance < 0.75: #Have 25% to get from APL2 if it exists. If it doesn't exist, so search for APL1
					if int(random.random()): #50% to get right/left APL-2.
						key = aa_cur + aa_nxt + '_' + ss_cur + ss_next
						if key in self.APL[self.protein+'-2-Right']:
							prob = self.APL[self.protein+'-2-Right'][key]
						else:
							key = aa_cur + '_' + ss_cur
							prob = self.APL[self.protein+'-1'][key]
					else:
						key = aa_prv + aa_cur + '_' + ss_prv + ss_cur
						if key in self.APL[self.protein+'-2-Left']:
							prob = self.APL[self.protein+'-2-Left'][key]
						else:
							key = aa_cur + '_' + ss_cur
							prob = self.APL[self.protein+'-1']
				else: #25% to get from APL1
					key = aa_cur + '_' + ss_cur
					prob = self.APL[self.protein+'-1'][key]
			else: #Using only APL-1
				prob = []
				aa_cur = self.primary_structure[i]
				ss_cur = self.secondary_structure[i]
				key = aa_cur + '_' + ss_cur
				prob = self.APL[self.protein+'-1'][key]

			aa_angles.append(self.use_histogram(prob))

		return aa_angles

	def use_histogram(self, prob_list):
		prob_radius = 0.5
		float_precision = 3
		id_probs = 0
		rnd = random.random()
		i = 0

		final = []

		for w in prob_list:
			rnd -=w[2]
			if rnd< 0:
				id_probs = i
				break
			i += 1

		probs = prob_list[id_probs]
		phi = probs[0] + random.uniform(-prob_radius, prob_radius)
		psi = probs[1] + random.uniform(-prob_radius, prob_radius)
		angles = random.choice(probs[3])
		omega = angles[0] + random.uniform(-prob_radius, prob_radius)

		final = [round(phi, float_precision), round(psi, float_precision), round(omega, float_precision)]

		return final

#objeto classe Histograma
#hist=HistogramFiles(PROTEIN_I, primary_amino_sequence, secondary_amino_sequence)
#hist = HistogramFiles('1ACW', 'VSCEDCPEHCSTQKAQAKCDNDKCVCEPI', 'CCCHHHHHHHHHTTTEEEEETTEEEEECC')
#final_anges = generate_first_angles(hist)
#print(final_anges)
