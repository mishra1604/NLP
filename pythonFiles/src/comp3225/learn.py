import re
import codecs
import nltk


# match = re.findall("[aA]","Hi I am Aaryan")

# print("match = '%s'" % match)

# regex for UK post code format
# regex = r"([A-Z]{1,2}[0-9][0-9A-Z]?)\s*([0-9][A-Z]{2})"
# match = re.search(regex, "SW1A 0AA")
# if match:
#     print("match = '%s'" % match.group(0))
#     print("match = '%s'" % match.group(1))
#     print("match = '%s'" % match.group(2))

# regex = r"([Country:])\s*([A-Z][a-z]+,\s*[A-Z][a-z]+)"
# regex to get country names from the text
pattern = r'''([C][o][u][n][t][r][y][:])\s*([A-Z][a-z]+,(\s*[A-Z]*[a-z]+)+)
        | ([A-Z]{1,2}[0-9][0-9A-Z]?)\s*([0-9][A-Z]{2})
        |'''

text = "Country: Alabama, United States of America"

tokens = nltk.regexp_tokenize(text, pattern)
print(tokens)