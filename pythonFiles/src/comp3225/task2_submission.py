# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/01/29
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

def exec_regex_questions( file_chapter = None ) :
    
	# CHANGE BELOW CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (task 2)

	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> questions.txt = plain text set of extracted questions. one line per question.

	# hardcoded output to show exactly what is expected to be serialized

    # question regex for a chapter in a book
    tester = r"\b\‘?[A-Z][\w\s\,\’\‘\;\“\”\-]*\?\’?"

    questionSentences = []
    matches = re.findall(tester, codecs.open(file_chapter,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        question = match
        question = question.replace("\r\n", " ")
        question = question.replace("\r", "")
        question = question.rstrip()
        questionSentences.append(question)
    

    filterRegex = r"\,\s\‘(.*)\’?"
    filter_2 = r"\;\s(.*)"
    filteredQuestions = []
    for match in questionSentences:
        filteredMatch = re.findall(filterRegex, match)
        if len(filteredMatch) == 0:
            filteredMatch = re.findall(filter_2, match)
        if filteredMatch:
            question = filteredMatch[0]
            if question[-1] == "’": question = question[:-1]
            if question[0] == "‘": question = question[1:]
            filteredQuestions.append(question)
        else:
            if match[-1] == "’": match = match[:-1]
            if match[0] == "‘": match = match[1:]
            filteredQuestions.append(match)
    
    setQuestions = set(filteredQuestions)

    # DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    writeHandle = codecs.open( 'questions.txt', 'w', 'utf-8', errors = 'replace' )
    for strQuestion in setQuestions :
        writeHandle.write( strQuestion + '\n' )
    writeHandle.close()

if __name__ == '__main__':
	if len(sys.argv) < 4 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book_file = sys.argv[2]
	chapter_file = sys.argv[3]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book = ' + repr(book_file) )
	logger.info( 'chapter = ' + repr(chapter_file) )

	# DO NOT CHANGE THE CODE IN THIS FUNCTION

	exec_regex_questions( chapter_file )