from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics


# extracting chapter headings from a book using regex and to build a table of contents

def exec_regex_toc(file_book = None):
    # regex for chapter headings format -> CHAPTER 1: THE BEGINNING
    chapter = r'(CHAPTER\s\d+\.)\s([A-Z\s]+)'

    # chapter number regex
    chapterNumberPattern = r'(\d+)'


    matchList = []
    for line in codecs.open(fname,"r",encoding="utf-8"):
        match = re.search(chapter, line)
        matchedTuple = []
        if match:
            chapterNumberMatch = re.search(chapterNumberPattern, match.group(0))
            if chapterNumberMatch:
                matchedTuple.append(chapterNumberMatch.group(0))
            matchedTuple.append(match.group(2)[:len(match.group(2))-2])
            matchList.append(matchedTuple)
    
    # converting to dictTOC format
    dictTOC = {}
    for i in matchList:
        dictTOC[i[0]] = i[1]
    
    return dictTOC


if __name__ == '__main__':
    fname="../../corpus/comp3225/eval_book.txt"
    print(exec_regex_toc(fname))