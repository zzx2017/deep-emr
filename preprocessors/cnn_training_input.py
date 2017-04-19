import itertools
import pandas
from os import listdir
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

def get_annotation(element, indicator):
	return (element.getAttribute('text').strip().lower(), element.tagName.lower() + '.' + 
			element.getAttribute(indicator).strip().replace(' ', '_'))

def find_sublist(sublist, alist):
	indices = list()
	for index in (i for i, e in enumerate(alist) if e == sublist[0]):
		if alist[index:index + len(sublist)] == sublist:
			indices.append((index, index + len(sublist) - 1))
	return indices

def annotate(tags, annotations, indices):
	for i in range(len(indices)):
		for j in range(len(indices[i])):
				for k in range(indices[i][j][0], indices[i][j][1] + 1):
					tags[k] = 'I-' + annotations[i][1]

def isplit(iterable, splitters):
    return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in splitters) if not k]

def replace_elements(alist, indices):
	for i in range(len(indices)):
		for j in range(len(indices[i])):
			alist[i][indices[i][j]] = -1

def print_data(encoded_data, encoded_labels, data_indices):
	for i in range(len(encoded_data)):
		for j in range(len(encoded_data[i])):
			for k in range(len(encoded_data[i][j])):
				print(data_indices[encoded_data[i][j][k] - 2][0] + " " + 
					(','.join(str(x) for x in encoded_labels[i][j])))

def write_to_file(filename, data, index):
	file = open(filename, 'w')
	for i in range(len(data)):
		file.write("%d %s\n" % (i + index, data[i][0]))
	file.close()

def generate_files(data, labels, files):
	dataset = open('dataset.txt', 'w')
	for i in range(0, len(data)):
		file = open('output/' + files[i][0:-4] + '.txt', 'w')
		for j in range(0, len(data[i])):
			file.write(','.join(str(x) for x in data[i][j]) + ' ' + (','.join(str(x) for x in labels[i][j])) + '\n')
			dataset.write(','.join(str(x) for x in data[i][j]) + ' ' + (','.join(str(x) for x in labels[i][j])) + '\n')
		file.close()
	dataset.close()

tagnames = ['CAD', 'DIABETES', 'FAMILY_HIST', 'HYPERLIPIDEMIA', 'HYPERTENSION', 'MEDICATION', 'OBESE', 'SMOKER']

classes = {'cad.mention': 1, 'cad.event': 2, 'cad.test': 3, 'cad.symptom': 4, 
'diabetes.mention': 5, 'diabetes.A1C': 6, 'diabetes.glucose': 7,
'family_hist.notpresent': 8, 'family_hist.present': 9,
'hyperlipidemia.mention': 10, 'hyperlipidemia.high_LDL': 11, 'hyperlipidemia.high_chol.': 12,
'hypertension.mention': 13, 'hypertension.high_bp': 14,
'medication.ACE_inhibitor': 15, 'medication.anti_diabetes': 16, 'medication.ARB': 17, 'medication.aspirin': 18,
'medication.beta_blocker': 19, 'medication.calcium_channel_blocker': 20, 'medication.diuretic': 21,
'medication.DPP4_inhibitors': 22, 'medication.ezetimibe': 23, 'medication.fibrate': 24, 'medication.insulin': 25,
'medication.metformin': 26, 'medication.niacin': 27, 'medication.nitrate': 28, 'medication.statin': 29, 
'medication.sulfonylureas': 30, 'medication.thiazolidinedione': 31, 'medication.thienopyridine': 32,
'obese.mention': 33, 'obese.BMI': 34, 
'smoker.current': 35, 'smoker.ever': 36, 'smoker.never': 37, 'smoker.past': 38,'smoker.unknown': 39}

files = [file for file in listdir('.') if file.endswith('.xml')]
data, data_list, labels, label_list = list(), list(), list(), list()

for file in files:
	root = minidom.parse(file)

	annotation_objects = [root.getElementsByTagName(x) for x in tagnames]
	annotations = [[[get_annotation(z, 'type1')
				if z.tagName == 'MEDICATION' else get_annotation(z, 'status')
				if z.tagName == 'SMOKER' else get_annotation(z, 'indicator')
				for z in y.getElementsByTagName(y.tagName)] 
				for y in x] for x in annotation_objects]
	annotations = [[y for y in x if len(y) > 0] for x in annotations if len(x) > 0]
	annotations = list(set([y for x in [y for x in annotations for y in x] for y in x]))
	annotations = [(word_tokenize(x[0]), x[1]) for x in annotations if x[0] != '']
	annotations.sort(key=lambda x: len(x[0]), reverse=True)
	
	text = root.getElementsByTagName("TEXT")[0].firstChild.data
	text = word_tokenize(text.lower())

	indices = [find_sublist(x[0], text) for x in annotations]
	tags = ['O' for x in text]
	annotate(tags, annotations, indices)
	data.extend(text)
	labels.extend(tags)
	data_list.append(text)
	label_list.append(tags)

data_indices = Counter(data).most_common()
label_indices = Counter(labels).most_common()

encoded_data = [[(i + 2) for y in x for i, a in enumerate(data_indices) if y == a[0]] for x in data_list]
encoded_labels = [[(i + 1) for y in x for i, a in enumerate(label_indices) if y == a[0]] for x in label_list]

period_index = [i + 2 for i, x in enumerate(data_indices) if x[0] == "."][0]
period_indices = [[i for i, y in enumerate(x) if y == period_index] for x in encoded_data]

encoded_data = [isplit(x, (period_index,)) for x in encoded_data]
replace_elements(encoded_labels, period_indices)
encoded_labels = [isplit(x, (-1,)) for x in encoded_labels]

encoded_labels = [[[classes[label_indices[z - 1][0][2::]] for z in list(set(y)) if z != 1] for y in x] for x in encoded_labels]
encoded_labels = [[[0] if len(y) == 0 else y for y in x] for x in encoded_labels]

write_to_file('dictionary.txt', data_indices, 2)
generate_files(encoded_data, encoded_labels, files)
