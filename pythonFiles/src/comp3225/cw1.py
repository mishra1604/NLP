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

def test_regex_toc():
    fname="../../corpus/comp3225/tale.txt"

    chapter = r'CHAPTER'
    roman = r'(?:[IVXLCDM]+)'
    chapter_sep = chapter + r"|\n\n"
    chapter_regex = r"^\s*" + chapter + r"\s*" + roman + r"\W\s*((?!" + chapter_sep + ")[\w\.\'\":]+)"

    self_pattern = r"CHAPTER\s+(\w+)\.(?:\r\n|\s*)(.*)"
    gpt_pattern = r"CHAPTER\s+(\w+)\.\s*(.*)"


    matches = re.findall(self_pattern, codecs.open(fname,"r",encoding="utf-8").read(), re.MULTILINE)
    toc = {}
    print("length of matches", len(matches))
    for match in matches:
        print("Chapter:", match[0])
        print("Title:", match[1])
        toc[match[0]] = match[1][:len(match[1])-1]
    
    print("length of toc", len(toc))
    


if __name__ == '__main__':
    test_regex_toc()