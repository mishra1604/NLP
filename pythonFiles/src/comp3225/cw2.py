from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )

def test_question_regex():
    fname="../../corpus/comp3225/eval_chapter.txt"

    # question regex for a chapter in a book
    question = r"(?:\.|\?|^)\s?(.?[\w+\s]+\?.?\s)" #(?:\.|\?)\s+(.*\?)\s+ 

    matches = re.findall(question, codecs.open(fname,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        print(match)

if __name__ == '__main__':
    test_question_regex()