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
    fname="../../corpus/comp3225/hard.txt"

    chapter = r'CHAPTER'
    roman = r'(?:[IVXLCDM]+)'
    chapter_sep = chapter + r"|\n\n"
    chapter_regex = r"^\s*" + chapter + r"\s*" + roman + r"\W\s*((?!" + chapter_sep + ")[\w\.\'\":]+)"

    self_pattern = r"^CHAPTER\s+(\w+)\.?(?:\r\n(?:\r\n)?|\s*)(.*)"
    second_pattern = r"^(\d+)\.\s+_(.*?\?*_)\s*$" #r"^(\d+)\.?\s+_([\w+\s])_\r\n"
    gpt_pattern = r"CHAPTER\s+(\w+)\.\s*(.*)"

    book_pattern =  r"^(Book|BOOK|PART|Part)\s*(?:the|THE)?\s*(\w+)" #r"Book\s(?:\w+)\s(/w+\-\-)"

    bookChapters = []
    matches = re.findall(book_pattern, codecs.open(fname,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        if match[0] == "PART":
            bookChapters.append("Part " + match[1])
        elif match[0] == "Book":
            bookChapters.append("Book " + match[1])
        elif match[0] == "BOOK":
            bookChapters.append("Book " + match[1])
        # bookChapters.append(match)
    
    
    toc = {}
    x = 0
    matches = re.findall(self_pattern, codecs.open(fname,"r",encoding="utf-8").read(), re.MULTILINE)
    if len(matches) == 0:
        matches = re.findall(second_pattern, codecs.open(fname,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        # print("Chapter:", match[0])
        # print("Title:", match[1])
        if len(bookChapters) >0:
            if "("+str(bookChapters[x])+")" + " " + match[0] in toc:
                x += 1
                toc["("+str(bookChapters[x])+")" + " " + match[0]] = match[1][:len(match[1])-1]
            else:
                toc["("+str(bookChapters[x])+")" + " " + match[0]] = match[1][:len(match[1])-1]
        else:
            toc[match[0]] = match[1][:len(match[1])-1]
    
    print(toc)
    


if __name__ == '__main__':
    test_regex_toc()