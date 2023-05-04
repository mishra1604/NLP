from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

display_label_subset = [ 'B-DATE', 'I-DATE', 'O', 'B-CARD', 'I-CARD', 'B-ORD', 'I-ORD', 'B-NORP', 'I-NORP', 'B-PERSON', 'I-PERSON' ]

def create_dataset( max_files = None, ontonotes_file = None ):
    # load parsed ontonotes dataset
    readHandle = codecs.open( ontonotes_file, 'r', 'utf-8', errors = 'replace' )
    str_json = readHandle.read()
    readHandle.close()
    dict_ontonotes = json.loads( str_json )

    # make a training and test split
    list_files = list( dict_ontonotes.keys() )
    if len(list_files) > max_files :
        list_files = list_files[ :max_files ]
    nSplit = math.floor( len(list_files)*0.9 )
    list_train_files = list_files[ : nSplit ]
    list_test_files = list_files[ nSplit : ]

    print("size of list_files: ", len(list_files))

    # sent = (tokens, pos, IOB_label)
    list_train = []
    for str_file in list_train_files :
        for str_sent_index in dict_ontonotes[str_file] :
            # ignore sents with non-PENN POS tags
            if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
                continue
            if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
                continue

            list_entry = []

            # compute IOB tags for named entities (if any)
            ne_type_last = None
            for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
                strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
                strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
                ne_type = None
                if 'ne' in dict_ontonotes[str_file][str_sent_index] :
                    dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
                    if not 'parse_error' in dict_ne :
                        for str_NEIndex in dict_ne :
                            if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
                                ne_type = dict_ne[str_NEIndex]['type']
                                break
                if ne_type != None :
                    if ne_type == ne_type_last :
                        strIOB = 'I-' + ne_type
                    else :
                        strIOB = 'B-' + ne_type
                else :
                    strIOB = 'O'
                ne_type_last = ne_type

                list_entry.append( ( strToken, strPOS, strIOB ) )

            list_train.append( list_entry )

    list_test = []
    for str_file in list_test_files :
        for str_sent_index in dict_ontonotes[str_file] :
            # ignore sents with non-PENN POS tags
            if 'XX' in dict_ontonotes[str_file][str_sent_index]['pos'] :
                continue
            if 'VERB' in dict_ontonotes[str_file][str_sent_index]['pos'] :
                continue

            list_entry = []

            # compute IOB tags for named entities (if any)
            ne_type_last = None
            for nTokenIndex in range(len(dict_ontonotes[str_file][str_sent_index]['tokens'])) :
                strToken = dict_ontonotes[str_file][str_sent_index]['tokens'][nTokenIndex]
                strPOS = dict_ontonotes[str_file][str_sent_index]['pos'][nTokenIndex]
                ne_type = None
                if 'ne' in dict_ontonotes[str_file][str_sent_index] :
                    dict_ne = dict_ontonotes[str_file][str_sent_index]['ne']
                    if not 'parse_error' in dict_ne :
                        for str_NEIndex in dict_ne :
                            if nTokenIndex in dict_ne[str_NEIndex]['tokens'] :
                                ne_type = dict_ne[str_NEIndex]['type']
                                break
                if ne_type != None :
                    if ne_type == ne_type_last :
                        strIOB = 'I-' + ne_type
                    else :
                        strIOB = 'B-' + ne_type
                else :
                    strIOB = 'O'
                ne_type_last = ne_type

                list_entry.append( ( strToken, strPOS, strIOB ) )

            list_test.append( list_entry )

    return list_train, list_test

def sent2features(sent, word2features_func = None):
	return [word2features_func(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, postag, label in sent]

def task2_word2features(sent, i):

    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'word' : word,
        'postag': postag,

        # token shape
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),

        # token suffix
        'word.suffix': word.lower()[-3:],

        # POS prefix
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word_prev = sent[i-1][0]
        postag_prev = sent[i-1][1]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:postag': postag_prev,
            '-1:word.lower()': word_prev.lower(),
            '-1:word.isupper()': word_prev.isupper(),
            '-1:word.istitle()': word_prev.istitle(),
            '-1:word.isdigit()': word_prev.isdigit(),
            '-1:word.suffix': word_prev.lower()[-3:],
            '-1:postag[:2]': postag_prev[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word_next = sent[i+1][0]
        postag_next = sent[i+1][1]
        features.update({
            '+1:word.lower()': word_next.lower(),
            '+1:postag': postag_next,
            '+1:word.lower()': word_next.lower(),
            '+1:word.isupper()': word_next.isupper(),
            '+1:word.istitle()': word_next.istitle(),
            '+1:word.isdigit()': word_next.isdigit(),
            '+1:word.suffix': word_next.lower()[-3:],
            '+1:postag[:2]': postag_next[:2],
        })
    else:
        features['EOS'] = True

    return features

def exec_task( max_files = None, max_iter = None, display_label_subset = [], word2features_func = None, train_crf_model_func = None, dataset_file = None ) :

    # make a dataset from english NE labelled ontonotes sents
    train_sents, test_sents = create_dataset( max_files = max_files, ontonotes_file = dataset_file)

    # create feature vectors for every sentence
    X_train = [sent2features(s, word2features_func = word2features_func) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s, word2features_func = word2features_func) for s in test_sents]
    Y_test = [sent2labels(s) for s in test_sents]

    final_training_set = X_train + X_test
    final_training_labels = Y_train + Y_test


    # getting the set of labels that exist in the sentences
    set_labels = set([])
    for data in [Y_train,Y_test] :
        for n_sent in range(len(data)) :
            for str_label in data[n_sent] :
                set_labels.add( str_label )
    labels = list( set_labels )
    # logger.info( 'labels = ' + repr(labels) )

    # remove 'O' label as we are not usually interested in how well 'O' is predicted
    #labels = list( crf.classes_ )
    labels.remove('O')

    # Train CRF model
    crf = train_crf_model_func( final_training_set, final_training_labels, max_iter, labels )
    
    return crf

def task1_train_crf_model( X_train, Y_train, max_iter, labels ) :
    # train the basic CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=max_iter,
        all_possible_transitions=True,
    )
    crf.fit(X_train, Y_train)
    return crf

def exec_ner( file_chapter = "../../corpus/comp3225/trial.txt", ontonotes_file = '../../corpus/comp3225/ontonotes_parsed.json' ):

    # open the file and get the sentences
    with open(file_chapter, 'r', encoding='utf8') as f:
        lines = f.readlines()

    chapter_paragraphs = []

    paragraph_string = ""
    for line in lines:
        main_string = line
        if len(main_string) == 1:
            if len(paragraph_string) > 0:
                chapter_paragraphs.append(paragraph_string)
                paragraph_string = ""
        else:
            main_string = main_string.replace("\n", " ")
            paragraph_string += main_string

    # create tokens for all the paragraphs in the chapter
    chapter_tokens = [nltk.word_tokenize(paragraph) for paragraph in chapter_paragraphs]

    # get the POS tags for the tokens
    chapter_pos_tags = [nltk.pos_tag(paragraph_tokens) for paragraph_tokens in chapter_tokens]

    # create a list of tuples for the tokens and POS tags
    chapter_tokens_pos_tags = [[(token, pos_tag) for token, pos_tag in paragraph_pos_tags] for paragraph_pos_tags in chapter_pos_tags]


    # get the chapter features
    chapter_features = [sent2features(paragraph_tokens_pos_tags, word2features_func = task2_word2features) for paragraph_tokens_pos_tags in chapter_tokens_pos_tags]

    # load the model
    crf_model = exec_task( word2features_func = task2_word2features, train_crf_model_func = task1_train_crf_model, max_files = 100, max_iter = 100, display_label_subset = display_label_subset, dataset_file = ontonotes_file )

    # predict the labels for every paragraph in chapter_features
    chapter_labels = [crf_model.predict_single(paragraph_features) for paragraph_features in chapter_features]

    rough_dictNE = {
        "PERSON": []
    }
    for paragraph_label in chapter_labels:
        counter = 0
        paragraph_tokens = chapter_tokens[chapter_labels.index(paragraph_label)]
        while counter < len(paragraph_label):
            if paragraph_label[counter] == "B-PERSON":
                person = ""
                if counter > 0 and (paragraph_tokens[counter - 1] == "Mr." or paragraph_tokens[counter - 1] == "Mrs." or paragraph_tokens[counter - 1] == "Ms."): 
                    person += paragraph_tokens[counter - 1] + " "
                person += paragraph_tokens[counter]
                counter += 1
                while counter < len(paragraph_label) and paragraph_label[counter] == "I-PERSON":
                    person += " " + paragraph_tokens[counter]
                    counter += 1
                if person not in rough_dictNE["PERSON"]:
                    rough_dictNE["PERSON"].append(person)
            else:   
                counter += 1

    print("Prescanning result",rough_dictNE)

    # regex for extracting names
    names_regex = r"(Mr.?|Mrs.?|Miss.?)? ?([A-Z][a-z]+) ([A-Z][a-z]+)?"

    # # scanning each person using nltk pos tagger and only selecting NNP or NNPS or NNS
    # for person in rough_dictNE["PERSON"]:
    #     person_tokens = nltk.word_tokenize(person)
    #     person_pos_tags = nltk.pos_tag(person_tokens)
    #     print(person_pos_tags)
    #     for token_pos_tag in person_pos_tags:
    #         if token_pos_tag[1] == "NNP" or token_pos_tag[1] == "NNPS" or token_pos_tag[1] == "NNS":
    #             continue
    #         else:
    #             rough_dictNE["PERSON"].remove(person)
    
    # dictNE = {
    #     "PERSON": []
    # }

    # for person in rough_dictNE["PERSON"]:
    #     refined_person = person.replace("’", "")
    #     refined_person = refined_person.replace("\"", "")
    #     refined_person = refined_person.replace("(", "")
    #     refined_person = refined_person.replace(")", "")
    #     refined_person = refined_person.rstrip()
    #     refined_person = refined_person.lower()
    #     dictNE["PERSON"].append(refined_person)
    

    print("\n\nFinal result", dictNE)

    



    answer = input("Do you want to show the labels? (y/n): ")
    if answer == "y": print("\n\nLabels:\n", [[(token, label) for token, label in zip(paragraph_tokens, paragraph_labels)] for paragraph_tokens, paragraph_labels in zip(chapter_tokens, chapter_labels)])



if __name__ == '__main__':
    exec_ner()