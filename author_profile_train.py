# -*- coding: utf-8 -*-

import sys
import pickle
from sklearn import preprocessing, linear_model
from sklearn.multiclass import OneVsRestClassifier
from train import Configuration, TrainingData, Train, label_extractor_age_gender, label_extractor_personality
from utils import detect_language, preprocess, tokenize,pre_process
import os
import re


def get_parser():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input",
                      help="path/to/training/corpus")
    parser.add_option("-o", "--output", dest="output",
                      help="path/to/output/directory")
    return parser


def save_label_encoder(le,path,lang,name):
    pickle.dump(le,open(os.path.join(path, lang, name), "w"))


def configure_age_gender(lang):
    """
    :param lang: language
    :return: best configuration of features for the given language
    """


    criteria = {
        'lang': lang,
        }

    if lang == 'en':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}

        return Configuration(criteria)
    if lang == 'es':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}
        return Configuration(criteria)
    if lang == 'it':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}
        return Configuration(criteria)
    if lang == 'nl':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}
        return Configuration(criteria)
    return Configuration(criteria)



def configure_personality(lang):
    criteria = {
        'lang': lang,
        }

    if lang == 'en':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}

        return Configuration(criteria)
    if lang == 'es':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}
        return Configuration(criteria)
    if lang == 'it':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}
        return Configuration(criteria)
    if lang == 'nl':
        criteria['twitter']=True
        criteria['lda']={'n_topics':8,'stop_words':'english'}
        criteria['categorical_char_ngram']={'ngram_range':(3, 3), 'preprocessor':pre_process, 'multi_word':True,'mid_word':True,'prefix':True}
        criteria['ngram']={'preprocessor':preprocess, 'tokenizer':tokenize, 'ngram_range':(1, 1),
                                      'min_df':2, 'analyzer':"word"}
        return Configuration(criteria)
    return Configuration(criteria)



def main(argv):
    parser = get_parser()
    (options, args) = parser.parse_args(argv)
    if not (options.input and options.output):
        parser.error("Requires arguments not provided")
    else:
        lang = detect_language(options.input)
        if lang not in ['en', 'es', 'it', 'nl']:
            print >> sys.stderr, 'Language other than en, es, it, nl'
            sys.exit(1)
        else:


            training_data = TrainingData(path=options.input).load_data()  # load training data
            #age gender
            configure_obj_ag = configure_age_gender(lang)  # apply which features to use
            label_encoder_ag = preprocessing.LabelEncoder()  # map feature name to integer index

            log_reg = linear_model.LogisticRegression(dual=True, multi_class='multinomial',
                                                      solver='lbfgs')  # multiclass classifier
            model_age_gender = Train(configuration=configure_obj_ag,
                          classifier=log_reg)  # create train pipeline


            #personality
            configure_obj_per=configure_personality(lang)

            label_encoder_per = preprocessing.MultiLabelBinarizer()
            # label_encoder_per = None

            linear_reg=OneVsRestClassifier(linear_model.LogisticRegression(dual=True))

            model_per = Train(configuration=configure_obj_per,
                          classifier=linear_reg)


            print "Training ......."

            model_age_gender.train(training_data.X,training_data.transform_label(label_extractor_age_gender,label_encoder_ag))
            model_per.train(training_data.X,training_data.transform_label(label_extractor_personality,label_encoder_per))

            print "Done "
            print "Saving model in {0} ".format(options.output)

            if not os.path.exists(os.path.join(options.output, lang)):
                os.mkdir(os.path.join(options.output, lang))

            model_age_gender.save(os.path.join(options.output, lang, 'age_gender.model'))
            model_per.save(os.path.join(options.output, lang, 'personality.model'))

            save_label_encoder(label_encoder_ag,options.output,lang,'age_gender.le')
            save_label_encoder(label_encoder_per,options.output,lang,'personality.le')

            print "Done saving model"


if __name__ == "__main__":
    main(sys.argv)


