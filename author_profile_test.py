# -*- coding: utf-8 -*-
import os
import sys
import pickle
from train import Test, label_extractor_gender_age, Model, label_extractor_personality_values
from utils import detect_language
import collections

def get_parser():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input",
                      help="path/to/training/corpus")
    parser.add_option("-m", "--model", dest="model",
                      help="path/to/model/directory")
    parser.add_option("-o", "--output", dest="output",
                      help="path/to/output/directory")
    return parser



def load_model(path,lang,name):
    age_gender_model= pickle.load(open(os.path.join(path,lang,name), "r"))
    return age_gender_model


def load_label_encoder(path, lang,name):
    age_gender_le= pickle.load(open(os.path.join(path,lang,name), "r"))
    return age_gender_le







def main(argv):
    parser = get_parser()
    (options, args) = parser.parse_args(argv)
    if not (options.input and options.output and options.model):
        parser.error("Requires arguments not provided")
    else:
        lang = detect_language(options.input)
        if lang not in ['en', 'es', 'it', 'nl']:
            print >> sys.stderr, 'Language other than en, es, it, nl'
            sys.exit(1)
        else:
            print "Loading model for lang {0}".format(lang)
            age_gender_model=load_model(options.model,lang,'age_gender.model')
            label_encoder_ag= load_label_encoder(options.model,lang,'age_gender.le')

            #personlality
            personality_model=load_model(options.model,lang,'personality.model')
            label_encoder_per=load_label_encoder(options.model,lang,'personality.le')

            print  "Done loading model"
            model_ag = Model(name='age_gender',clf=age_gender_model, label_encoder=label_encoder_ag, label_extractor=label_extractor_gender_age)
            model_per = Model(name='personality',clf=personality_model, label_encoder=label_encoder_per, label_extractor=label_extractor_personality_values)

            test =Test(model=[model_ag,model_per],path=options.input)
            test.run()
            test.save(options.output)

            print "Done "






if __name__=="__main__":
    main(sys.argv)
