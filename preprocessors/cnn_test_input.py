import numpy
import pandas
from os import listdir
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize

def get_annotation(element, indicator):
	return (element.getAttribute('text').strip().lower(), element.tagName.lower() + '.' + 
			element.getAttribute(indicator).strip().replace(' ', '_'))

def write_text(filename, data):
	file = open(filename, 'w')
	for i in range(0, len(data)):
		file.write(",".join(str(x) for x in data[i]) + '\n')
	file.close()

def write_annotations(filename, data):
	file = open(filename, 'w')
	file.write(",".join(str(x) for x in data))
	file.close()

tagnames = ['CAD', 'DIABETES', 'FAMILY_HIST', 'HYPERLIPIDEMIA', 'HYPERTENSION', 'MEDICATION', 'OBESE', 'SMOKER']

dictionary = pandas.read_csv("dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
classes = pandas.read_csv("classes.txt", delim_whitespace=True, header=None)
classes = classes.set_index(1)[0].to_dict()

files = [file for file in listdir('.') if file.endswith('.xml')]

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
	annotations = list(set([x[1] for x in annotations]))
	encoded_annotations = [classes[x] for x in annotations]
	encoded_annotations.sort(key=lambda x: x)

	text = root.getElementsByTagName("TEXT")[0].firstChild.data
	sentences = sent_tokenize(text)
	words = [[y.lower() for y in word_tokenize(x)] for x in sentences]
	encoded_words = [[dictionary[y] if (y in dictionary) else 1 for y in x] for x in words]

	write_text('output/text/' + file[0:-4] + '.txt', encoded_words)
	write_annotations('output/labels/' + file[0:-4] + '-labels.txt', encoded_annotations)
