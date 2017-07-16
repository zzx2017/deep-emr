import numpy
import pandas
import glob
import gensim
import multiprocessing
import logging
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

files = glob.glob("../data/complete/*.xml")
sentences = list()

for file in files:
    root = minidom.parse(file)
    text = root.getElementsByTagName("TEXT")[0].firstChild.data
    text = sent_tokenize(text)
    stemmer = SnowballStemmer("english")
    text = [[stemmer.stem(y.lower()) for y in word_tokenize(x)] for x in text]
    sentences.extend(text)

cores = multiprocessing.cpu_count()

logging.info('Training word2vec model')
model = gensim.models.Word2Vec(sentences=sentences, size=20, min_count=1, window=5, workers=cores)

logging.info('Saving model')
model.save('../models/word2vec/word2vec.model')
logging.info('Completed training word2vec model')
