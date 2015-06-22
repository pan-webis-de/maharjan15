# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
import twokenize
import string
import re
import numpy as np


#REF: Building Machine Learning Systems With Python book
emo_repl = {
    # positive emoticons
    "&lt;3": "emoticon_good",
    ":d": "emoticon_good",  # :D in lower case
    ":dd": "emoticon_good",  # :DD in lower case
    "8)": "emoticon_good",
    ":-)": "emoticon_good",
    ":)": "emoticon_good",
    ";)": "emoticon_good",
    "(-:": "emoticon_good",
    "(:": "emoticon_good",
    # negative emoticons:
    ":/": " emoticon_bad",
    ":&gt;": "emoticon_bad",
    ":')": "emoticon_bad",
    ":-(": "emoticon_bad",
    ":(": "emoticon_bad",
    ":S": "emoticon_bad",
    ":-S": "emoticon_bad",
}

emo_repl_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k
                                                       in emo_repl.keys()]))]

re_repl = {
    r"\n|\r": " ",
    r"((www\.[^\s]+)|(https?:\/\/[^\s]+))": "URL",  # url
    # r"#([^\s]+)": r"\1",  # remove hashtag
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not"
}


def preprocess(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    for r, repl in re_repl.iteritems():
        tweet = re.sub(r, repl, tweet)

    for k in emo_repl_order:
        tweet = tweet.replace(k, emo_repl[k])

    # remove punctuations
    # tweet=''.join(char for char in tweet if char not in string.punctuation)
    # trim
    tweet = tweet.strip('\'"')
    return tweet


def tokenize(tweet):
    return twokenize.tokenize(twokenize.normalizeTextForTagger(tweet))


# pan15-author-profiling-training-dataset-italian-2015-03-02
def detect_language(path, ):
    lang = ''
    for file in os.listdir(path):
        if file == 'truth.txt' or file == '.DS_Store':
            continue

        tree = ET.parse(os.path.join(path, file))
        root = tree.getroot()
        lang = root.get('lang')
        break
    return lang.strip()


def pre_process(tweet):
                tweet = tweet.lower()
                tweet = re.sub(r"((www\.[^\s]+)|(https?:\/\/[^\s]+))", "URL", tweet)
                return tweet



def rms_error(Y_actual,Y_predicted):
    Error = Y_actual-Y_predicted
    total_error = np.sum(Error*Error) # sum of squares
    rmse = np.sqrt(total_error*1.0/len(Y_actual))
    return rmse


def change_range(x, max_source, min_source, max_target, min_target):
   return min_target + x * (max_target - min_target) / (max_source - min_source)
