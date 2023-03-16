from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )

def test_question_regex():
    fname="../../corpus/comp3225/trial.txt"

    # question regex for a chapter in a book
    question = r"[\.\?^\,\!]\s?(.?[\w+\s\,\’\‘\-]+\?.?)" #(?:\.|\?|^|\,|\!)\s?(.?[\w+\s\,\’\‘\-]+\?.?)--original   (?:\.|\?|^|\,|\!)((\s?.?[\w+\s\,\’\‘\-]+\?.?)+)--new
    tester = r"\b\‘?[A-Z][\w\s\,\’\‘\;\“\”\-]*\?\’?"       #\b[A-Z][\w\s\,\’\‘\-]*\?

    questionSentences = []
    matches = re.findall(tester, codecs.open(fname,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        question = match
        question = question.replace("\r\n", " ")
        question = question.replace("\r", "")
        question = question.rstrip()
        questionSentences.append(question)
    
    # print(questionSentences, "\nlenght of question sentences:", len(questionSentences))

    filterRegex = r"\,\s\‘(.*)\’?"
    filter_2 = r"\;\s(.*)"
    filteredQuestions = []
    for match in questionSentences:
        filteredMatch = re.findall(filterRegex, match)
        if len(filteredMatch) == 0:
            filteredMatch = re.findall(filter_2, match)
        if filteredMatch:
            print(filteredMatch)
            question = filteredMatch[0]
            if question[-1] == "’": question = question[:-1]
            if question[0] == "‘": question = question[1:]
            filteredQuestions.append(question)
        else:
            if match[-1] == "’": match = match[:-1]
            if match[0] == "‘": match = match[1:]
            filteredQuestions.append(match)
    
    print(filteredQuestions, "\nlenght:", len(filteredQuestions))

    

if __name__ == '__main__':
    test_question_regex()
#     text = """‘Well Piggywiggies,’ said R. W., ‘how de do to-night? What I was
# thinking of, my dear,’ to Mrs Wilfer already seated in a corner with
# folded gloves, ‘was, that as we have let our first floor so well, and as
# we have now no place in which you could teach pupils even if pupils--’"""
#     questions = re.findall(r"\b\‘?[A-Z][\w\s\,\’\‘\;\“\”\-\.]*\?\’?", text)
#     print(questions)