import glob
import numpy
import pandas
import itertools
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

def get_annotation(element, indicator):
    if element.tagName == 'SMOKER' or element.tagName == 'FAMILY_HIST':
        return (element.tagName.lower() + '.' + 
                element.getAttribute(indicator).lower().strip().replace(' ', '_'))
    else:
        return (element.tagName.lower() + '.' + 
                element.getAttribute(indicator).lower().strip().replace(' ', '_'),  
                element.getAttribute('time').lower().strip().replace(' ', '_'))
    
def combine_annotations(annotations):
    results = list()
    for annotation in annotations:
        if len(annotation) == 2:
            if ((annotation[0], 'before_dct') in annotations and 
                (annotation[0], 'during_dct') in annotations and 
                (annotation[0], 'after_dct') in annotations):
                 results.append((annotation[0] + '.continuing'))
            else:
                results.append((annotation[0] + '.' + annotation[1]))
        else:
            results.append(annotation)
    return list(set(results))

def write_text(filename, data):
    file = open(filename, 'w')
    for i in range(0, len(data)):
        file.write(','.join(str(x) for x in data[i]) + '\n')
    file.close()

def write_annotations(filename, data):
    file = open(filename, 'w')
    if len(data) == 0:
        file.write('0')
    else:
        file.write(','.join(str(x) for x in data))
    file.close()

def write_to_file(filename, text, labels):
    file = open(filename, 'w')
    for i in range(0, len(text)):
        file.write(','.join(str(x) for x in text[i]) + '\n')
    if len(labels) == 0:
        file.write('0')
    else:
        file.write(','.join(str(x) for x in labels))
    file.close()
        
tagnames = ['CAD', 'DIABETES', 'FAMILY_HIST', 'HYPERLIPIDEMIA', 'HYPERTENSION', 'MEDICATION', 'OBESE', 'SMOKER']

dictionary = pandas.read_csv("../data/dictionary.txt", delim_whitespace=True, header=None)
dictionary = dictionary.set_index(1)[0].to_dict()
classes = pandas.read_csv("../data/classes.txt", delim_whitespace=True, header=None)
classes = classes.set_index(1)[0].to_dict()

files = glob.glob("../data/test/*.xml")

for file in files:

    root = minidom.parse(file)
    annotation_objects = [root.getElementsByTagName(x) for x in tagnames]
    annotations = [x for x in annotation_objects]
    annotations = [y for x in annotations for y in x]
    annotations = [get_annotation(x, 'type1')
                    if x.tagName == 'MEDICATION' else get_annotation(x, 'status')
                    if x.tagName == 'SMOKER' else get_annotation(x, 'indicator') 
                    for x in annotations]
    annotations = combine_annotations(annotations)
    annotations = [x for x in annotations if x != 'family_hist.not_present' and x != 'smoker.unknown']
    encoded_annotations = [classes['I-' + x] for x in annotations if ('I-' + x) in classes]
    encoded_annotations.sort(key=lambda x: x)
    
    text = root.getElementsByTagName("TEXT")[0].firstChild.data
    sentences = sent_tokenize(text)
    stemmer = SnowballStemmer("english")
    words = [[stemmer.stem(y.lower()) for y in word_tokenize(x)] for x in sentences]
    encoded_words = [[dictionary[y] if (y in dictionary) else 1 for y in x] for x in words]

    paths = ['../models/cnn/data/test/gold/', '../models/rnn/data/test/gold/', '../models/lstm/data/test/gold/', '../models/blstm/data/test/gold/']

    for i in range(len(paths)):
        write_to_file(paths[i] + file[13:-4] + '.txt', encoded_words, encoded_annotations)
