import glob
import itertools
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import Counter

def get_annotation(element, indicator):
    if element.tagName == 'SMOKER' or element.tagName == 'FAMILY_HIST':
        return (element.getAttribute('text').strip().lower(), element.tagName.lower() + '.' + 
                element.getAttribute(indicator).lower().strip().replace(' ', '_'))
    else:
        return (element.getAttribute('text').strip().lower(), element.tagName.lower() + '.' + 
                element.getAttribute(indicator).lower().strip().replace(' ', '_'), 
                element.getAttribute('time').lower().strip().replace(' ', '_'))
    
def tokenise_annotation(annotation):
    return (word_tokenize(annotation[0]), annotation[1])

def combine_annotations(annotations):
    types = list()
    results = list()
    for annotation in annotations:
        if len(annotation) == 3:
            types.append((annotation[1], annotation[2]))
    for annotation in annotations:
        if len(annotation) == 3:
            if ((annotation[1], 'before_dct') in types and 
                (annotation[1], 'during_dct') in types and 
                (annotation[1], 'after_dct') in types):
                 results.append((annotation[0], annotation[1] + '.continuing'))
            else:
                results.append((annotation[0], annotation[1] + '.' + annotation[2]))
        else:
            results.append(annotation)
    return list(set(results))

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
            
def write_to_file(filename, data, index):
    file = open(filename, 'w')
    for i in range(len(data)):
        file.write("%d %s\n" % (i + index, data[i][0]))
    file.close()

def generate_files(data, labels, files):
    paths = ['../models/cnn/data/training/', '../models/rnn/data/training/', '../models/lstm/data/training/']
    for i in range(0, len(data)):
        for path in paths:
            file = open(path + files[i][17:-4] + '.txt', 'w')
            for j in range(0, len(data[i])):
                file.write(','.join(str(x) for x in data[i][j]) + ' ' + (','.join(str(x) for x in labels[i][j])) + '\n')
            file.close()
    
def print_data(encoded_data, encoded_labels, data_indices, label_indices):
    for i in range(len(encoded_data)):
        for j in range(len(encoded_data[i])):
            for k in range(len(encoded_data[i][j])):
                print(data_indices[encoded_data[i][j][k] - 2][0] + " " + 
                    label_indices[encoded_labels[i][j][k] - 1][0])

tagnames = ['CAD', 'DIABETES', 'FAMILY_HIST', 'HYPERLIPIDEMIA', 'HYPERTENSION', 'MEDICATION', 'OBESE', 'SMOKER']

files = glob.glob("../data/complete/*.xml")
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
    annotations = [x for x in annotations if x[1] != 'family_hist.not_present' and x[1] != 'smoker.unknown']
    annotations = [x for x in annotations if x[0] != '']
    annotations = combine_annotations(annotations)
    annotations = [tokenise_annotation(x) for x in annotations]
    annotations.sort(key=lambda x: len(x[0]), reverse=True)
    
    text = root.getElementsByTagName("TEXT")[0].firstChild.data
    text = word_tokenize(text.lower())
    
    indices = [find_sublist(x[0], text) for x in annotations]
    tags = ['O' for x in text]
    annotate(tags, annotations, indices)
    
    stemmer = SnowballStemmer("english")
    text = [stemmer.stem(x) for x in text]
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

print_data(encoded_data, encoded_labels, data_indices, label_indices)

write_to_file('../data/dictionary.txt', data_indices, 2)
write_to_file('../data/classes.txt', label_indices, 1)
generate_files(encoded_data, encoded_labels, files)