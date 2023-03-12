from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')


def exec_regex_toc( file_book = None ):

    # CHANGE BELOW CODE TO USE REGEX TO BUILD A TABLE OF CONTENTS FOR A BOOK (task 1)

	# Input >> www.gutenberg.org sourced plain text file for a whole book
	# Output >> toc.json = { <chapter_number_text> : <chapter_title_text> }

	# hardcoded output to show exactly what is expected to be serialized

    self_pattern = r"^(?:CHAPTER|STAVE|Chapter)\s(\w+)\.?(?:\r\n(?:\r\n)?|\s*)((.*\s{2})+)"
    second_pattern = r"^(\d+)\.\s+(_.*?\?*_)\s*$"
    third_pattern = r"^([IVXLCDM]+)\s+(.*)"

    book_pattern =  r"^(Book|BOOK|PART|Part|VOLUME)\s*(?:the|THE)?\s*(\w+)"

    bookChapters = []
    matches = re.findall(book_pattern, codecs.open(file_book,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        if match[0] == "PART":
            bookChapters.append("PART " + match[1])
        elif match[0] == "Part":
            bookChapters.append("Part " + match[1])
        elif match[0] == "Book":
            bookChapters.append("Book " + match[1])
        elif match[0] == "BOOK":
            bookChapters.append("Book " + match[1])
        elif match[0] == "VOLUME":
            bookChapters.append("VOLUME " + match[1])
    
    
    dictTOC = {}
    x = 0
    matches = re.findall(self_pattern, codecs.open(file_book,"r",encoding="utf-8").read(), re.MULTILINE)
    if len(matches) == 0:
        matches = re.findall(second_pattern, codecs.open(file_book,"r",encoding="utf-8").read(), re.MULTILINE)
    if len(matches) == 0:
        matches = re.findall(third_pattern, codecs.open(file_book,"r",encoding="utf-8").read(), re.MULTILINE)
    for match in matches:
        # print("Chapter:", match[0])
        # print("Title:", match[1])
        chapterTitle = match[1]
        chapterTitle = chapterTitle.replace("\r\n", " ")
        chapterTitle = chapterTitle.replace("\r", "")
        chapterTitle = chapterTitle.rstrip()
        if len(bookChapters) >0:
            if "("+str(bookChapters[x])+")" + " " + match[0] in dictTOC:
                x += 1
                dictTOC["("+str(bookChapters[x])+")" + " " + match[0]] = chapterTitle
            else:
                dictTOC["("+str(bookChapters[x])+")" + " " + match[0]] = chapterTitle
        else:
            dictTOC[match[0]] = chapterTitle
    
	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

    writeHandle = codecs.open( 'toc.json', 'w', 'utf-8', errors = 'replace' )
    strJSON = json.dumps( dictTOC, indent=2 )
    writeHandle.write( strJSON + '\n' )
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
    exec_regex_toc( book_file )