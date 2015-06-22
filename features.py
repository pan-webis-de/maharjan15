# -*- coding: utf-8 -*-
from gensim import corpora, models
from gensim.models import Phrases
from nltk.corpus import stopwords
from sklearn import preprocessing

from sklearn.base import BaseEstimator
import numpy as np

import re
from sklearn.feature_extraction.text import TfidfVectorizer, VectorizerMixin
from model import Document
from twokenize import tokenizeRawTweetText
import string
import nltk
import twokenize
import utils
import codecs
import os


def read_topic_tokens(path, name):
    token = {}
    with codecs.open(path, mode='rb') as f:
        for line in f.readlines():
            line = line.rstrip('\r\n')
            topic_words = line.split(',')
            token[name + '_' + topic_words[0]] = set(
                topic_words[1:])  # trim white spaces make sure in file no white space in tokens
            # print liwc
    return token


class NGramTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.content))


class FamilialFeatures(BaseEstimator):
    def __init__(self, lang):
        self.lang = lang
        self.male, self.female, self.neutral = self._initialize()
        # self.male = set([u'wife', u'gf', u'girlfriend', u'dw'])
        # self.female = set([u'husband', u'bf', u'boyfriend', u'dh', u'hubby'])
        # self.neutral = set([u'ds', u'dd', u'son', u'daughter', u'grandson', u'granddaughter', u'father',
        # u'mother', u'brother', u'sister', u'uncle', u'aunt', u'cousin', u'nephew', u'niece',
        # u'family', u'godson', u'goddaughter', u'grandchild', u'grandmother', u'grandfather',
        # u'baby', u'babies', u'child', u'children', u'kids', u'mom', u'parent'])

    def _initialize(self):
        token_dict = read_topic_tokens(os.path.join('resources', 'familial_tokens', self.lang + '.txt'), 'FT')
        #print token_dict
        return token_dict['FT_Male'], token_dict['FT_Female'], token_dict['FT_Neutral']

    def get_feature_names(self):
        return np.array(['male_bucket', 'female_bucket', 'neutral_bucket'])


    def fit(self, documents, y=None):
        return self


    def transform(self, documents):

        if self.lang == 'en':
            my = 'my'
        elif self.lang == 'es':
            my = 'mi'
        elif self.lang == 'it':
            my = 'il mio'
        else:
            my = 'mijn'

        def regex_str(items):
            "\b(?:DH|DS|DD|DW)\b"
            # regex_1 = r'(\b(?:' + '|'.join(items & set(['dw', 'ds', 'dd', 'dh'])) + ')\b)'
            # regex_2 = r'(\bmy\b\W+\b' + '(?:' + '|'.join(items - set(['dw', 'ds', 'dd', 'dh'])) + ')\W*\b)'
            # return twokenize.regex_or(regex_1,regex_2)


            abbrs = items & set(['dw', 'ds', 'dd', 'dh'])
            abbrs_joined = '|'.join(abbrs) + "|" if len(abbrs) else ""
            fulls_joined = '|'.join(items - set(['dw', 'ds', 'dd', 'dh']))
            my_regex = r'\b(?:' + abbrs_joined + my + r'\b\W+\b(?:' + fulls_joined + r'))\W*'
            return my_regex

        # total number of words
        male = re.compile(regex_str(self.male), re.IGNORECASE)
        # male = re.compile(r"\b(?:dw|my\b\W+\b(?:girlfriend|gf|wife))\W*", re.IGNORECASE)
        # print regex_str(self.male)
        female = re.compile(regex_str(self.female), re.IGNORECASE)
        neutral = re.compile(regex_str(self.neutral), re.IGNORECASE)

        # n_words = [len(tokenizeRawTweetText(doc)) for doc in documents]
        # print n_words
        #print [doc.content for doc in documents]
        male_count = [len(male.findall(doc.content.lower()))
                      for doc in documents]
        # print male_count
        female_count = [len(female.findall(doc.content.lower()))
                        for doc in documents]
        # print female_count
        neutral_count = [len(neutral.findall(doc.content.lower()))
                         for doc in documents]
        # print neutral_count
        male_bucket = np.array(male_count)
        female_bucket = np.array(female_count)
        neutral_bucket = np.array(neutral_count)
        X = np.array([male_bucket, female_bucket, neutral_bucket]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)


class TwitterFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(
            ['n_words', 'n_chars', 'numtexttoken', 'allcaps', 'exclamation', 'question', 'hashtag', 'mentioning',
             'urls', 'avgwordlenght', 'avetweetlength'])


    def fit(self, documents, y=None):
        return self


    def transform(self, documents):
        user_mentions = re.compile(ur"@\w\w+:?")
        text_number = re.compile(ur"\b\w*\d+\w*\b")
        url = re.compile(ur"((www\.[^\s]+)|(https?:\/\/[^\s]+))")
        mentioning = [len(user_mentions.findall(doc.content)) for doc in documents]
        # print mentioning
        exclamation = [doc.content.count('!') for doc in documents]
        # print exclamation
        question = [doc.content.count('?') for doc in documents]
        # print question
        hashtag = [doc.content.count('#') for doc in documents]
        # print hashtag
        n_words = [len(tokenizeRawTweetText(doc.content)) for doc in documents]
        # print n_words
        n_chars = [len(doc.content) for doc in documents]

        avetweetlength = [np.mean([len(tweet) for tweet in nltk.sent_tokenize(doc.content)]) for doc in documents]
        # print avetweetlength
        avgwordlenght = [np.mean([len(word) for word in tokenizeRawTweetText(doc.content)]) for doc in documents]
        # print avgwordlenght
        allcaps = [np.sum([word.isupper() for word in tokenizeRawTweetText(doc.content)]) for doc in documents]
        # print allcaps
        numtexttoken = [len(text_number.findall(doc.content)) for doc in documents]
        # print numtexttoken
        url_count = [len(url.findall(doc.content)) for doc in documents]
        # print url_count

        X = np.array([n_words, n_chars, numtexttoken, allcaps, exclamation, question, hashtag, mentioning, url_count,
                      avgwordlenght, avetweetlength]).T
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)
        return self.scalar.transform(X)


class CategoricalCharNgramsVectorizer(TfidfVectorizer):
    _slash_W = string.punctuation + " "

    _punctuation = r'''['\"“”-‘’.?!…,:;#\<\=\>@\(\)\*]'''
    _beg_punct = lambda self, x: re.match('^' + self._punctuation + '\w+', x)
    _mid_punct = lambda self, x: re.match(r'\w+' + self._punctuation + '(?:\w+|\s+)', x)
    _end_punct = lambda self, x: re.match(r'\w+' + self._punctuation + '$', x)

    # re.match is anchored at the beginning
    _whole_word = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (
        i == 0 or y[i - 1] in self._slash_W) and (i + n == len(y) or y[i + n] in self._slash_W)
    _mid_word = lambda self, x, y, i, n: not (
        re.findall(r'(?:\W|\s)', x) or i == 0 or y[i - 1] in self._slash_W or i + n == len(y) or y[
            i + n] in self._slash_W)
    _multi_word = lambda self, x: re.match('\w+\s\w+', x)

    _prefix = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (i == 0 or y[i - 1] in self._slash_W) and (
        not (i + n == len(y) or y[i + n] in self._slash_W))
    _suffix = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (
        not (i == 0 or y[i - 1] in self._slash_W)) and (i + n == len(y) or y[i + n] in self._slash_W)
    _space_prefix = lambda self, x: re.match(r'''^\s\w+''', x)
    _space_suffix = lambda self, x: re.match(r'''\w+\s$''', x)

    # def _whole_word(self,x, y, i, n):
    # if i == 0 or y[i-1] in self._slash_W:
    # if i + n == len(y) or y[i+n] in self._slash_W:
    # print "here 2"
    # return True
    # return False
    #
    # def _mid_word(self,x, y, i, n):
    # if i == 0 or y[i-1] in self._slash_W or i + n == len(y) - 1 or y[i+n] in self._slash_W:
    # return False
    # return True

    def __init__(self, beg_punct=None, mid_punct=None, end_punct=None, whole_word=None, mid_word=None, multi_word=None,
                 prefix=None, suffix=None, space_prefix=None, space_suffix=None, all=None, **kwargs):

        super(CategoricalCharNgramsVectorizer, self).__init__(**kwargs)
        self.beg_punct = beg_punct
        self.mid_punct = mid_punct
        self.end_punct = end_punct
        self.whole_word = whole_word
        self.mid_word = mid_word
        self.multi_word = multi_word
        self.prefix = prefix
        self.suffix = suffix
        self.space_prefix = space_prefix
        self.space_suffix = space_suffix
        self.all = all
        if self.all:
            self.beg_punct = True
            self.mid_punct = True
            self.end_punct = True
            self.whole_word = True
            self.mid_word = True
            self.multi_word = True
            self.prefix = True
            self.suffix = True
            self.space_prefix = True
            self.space_suffix = True


    # def _get_word(self, text_document, i, n):
    # start, end = 0, len(text_document)
    # for j in xrange(i, -1, -1):
    # if text_document[j] == ' ' or text_document[j] in string.punctuation:
    # start = j + 1
    # break
    #
    #     for k in xrange(i + n, len(text_document)):
    #         if text_document[k] == ' ' or text_document[k] in string.punctuation:
    #             end = k
    #             break
    #
    #     return text_document[start: end]

    def _categorical_char_ngrams(self, text_document):
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        # print min_n,max_n
        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                # check categories
                gram = text_document[i: i + n]
                added = False


                # punctuations
                if self.beg_punct and not added:
                    if self._beg_punct(gram):
                        ngrams.append(gram)
                        added = True

                if self.mid_punct and not added:
                    if self._mid_punct(gram):
                        ngrams.append(gram)
                        added = True

                if self.end_punct and not added:
                    if self._end_punct(gram):
                        ngrams.append(gram)
                        added = True


                # words


                if self.multi_word and not added:
                    if self._multi_word(gram):
                        ngrams.append(gram)
                        added = True

                if self.whole_word and not added:
                    if self._whole_word(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                if self.mid_word and not added:
                    if self._mid_word(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True


                #affixes
                if self.space_prefix and not added:
                    if self._space_prefix(gram):
                        ngrams.append(gram)
                        added = True

                if self.space_suffix and not added:
                    if self._space_suffix(gram):
                        ngrams.append(gram)
                        added = True

                if self.prefix and not added:
                    if self._prefix(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                if self.suffix and not added:
                    if self._suffix(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

        return ngrams


    def build_analyzer(self):
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._categorical_char_ngrams(preprocess(self.decode(doc.content)))


class LDA(BaseEstimator, VectorizerMixin):
    def __init__(self, **kwargs):
        self.number_of_topics = kwargs.get("n_topics", 10)
        self.preprocessor = kwargs.get("preprocessor", utils.preprocess)
        self.tokenizer = kwargs.get("tokenizer", utils.tokenize)

        self.stop_words = kwargs.get("stop_words", 'english')
        self.bigram = kwargs.get("bigram", None)
        self.trigram = kwargs.get("trigram", None)
        self.analyzer = kwargs.get("analyzer", 'word')
        self.ngram_range = kwargs.get('ngram_range', (1, 1))
        self.input = kwargs.get('input', 'content')
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.strip_accents = kwargs.get('strip_accents', None)
        self.decode_error = kwargs.get('decode_error', 'strict')
        self.lowercase = kwargs.get('lowercase', True)
        self.token_pattern = kwargs.get('token_pattern', r"(?u)\b\w\w+\b")


    def get_stop_words(self):
        stop_lst = stopwords.words(self.stop_words)
        # stop_lst=[]
        # if self.stop == "english":
        # stop_lst = stopwords.words('english')

        # print stop_lst

        stop_lst.extend(["@username", "URL", "i'm", "rt"])
        stop_lst.extend(list(string.punctuation))
        # print "Stopword list------->",stop_lst
        return stop_lst

    def get_feature_names(self):
        return np.array(
            ["Topic_" + str(topic) for topic in xrange(0, self.number_of_topics)])


    def _build_vocabulary(self, raw_documents, fixed_vocab=None):

        analyze = self.build_analyzer()
        doc_token_lst = [analyze(document.content) for document in raw_documents]

        if self.bigram or self.trigram:
            bigram = Phrases(doc_token_lst)

        if self.trigram:
            trigram = []

        if fixed_vocab:
            vocabulary = self.vocabulary_
            tfidf_model = self.tfidf_
            corpus_vector = [vocabulary.doc2bow(text) for text in doc_token_lst]
        else:
            vocabulary = corpora.Dictionary(doc_token_lst)

            corpus_vector = [vocabulary.doc2bow(text) for text in doc_token_lst]

            tfidf_model = models.TfidfModel(corpus_vector)
            self.tfidf_ = tfidf_model

        return tfidf_model[corpus_vector], vocabulary

    def fit(self, documents, y=None):
        self.fit_transform(documents)
        return self


    def fit_transform(self, raw_documents, y=None):

        # analyze = self.build_analyzer()
        # doc_token_lst = [analyze(document) for document in raw_documents]  # bigrams of phrases
        # if self.bigram or self.trigram:
        # bigram = Phrases(doc_token_lst)
        #
        # if self.trigram:
        # trigram = []
        #
        # vocabulary = corpora.Dictionary(doc_token_lst)
        # self.corpus_vector_ = [self.dictionary.doc2bow(text) for text in doc_token_lst]
        #
        # self.tfidf_ = models.TfidfModel(self.corpus_vector_)
        # self.corpus_tfidf_ = self.tfidf_[self.corpus_vector_]



        X, vocabulary = self._build_vocabulary(raw_documents, False)
        self.vocabulary_ = vocabulary

        self.lda_ = models.LdaModel(X, id2word=self.vocabulary_, num_topics=self.number_of_topics)

        # Topic_X=[]
        # for doc in X:
        # weight = np.zeros(self.number_of_topics)
        # for topic_id, prob in self.lda_[doc]:
        # weight[topic_id] = prob
        # Topic_X.append(weight)

        return self._fit_doc_topic(X)


    def _fit_doc_topic(self, X):
        Topic_X = []
        for doc in X:
            weight = np.zeros(self.number_of_topics)
            for topic_id, prob in self.lda_[doc]:
                weight[topic_id] = prob
            Topic_X.append(weight)
        # print Topic_X

        return np.array(Topic_X)


    def topics(self, n_topic_words, out_file):
        with codecs.open(out_file, 'w') as out:
            for i in range(0, self.number_of_topics):
                topic_words = [term[1].encode('utf-8') for term in self.lda_.show_topic(i, n_topic_words)]
                out.write("Top {} terms for topic #{} : {}".format(n_topic_words, i, ", ".join(topic_words)))
                out.write("\n\n================================================================================\n")


    def transform(self, documents):
        # if not hasattr(self, 'vocabulary_'):
        # self._check_vocabulary()

        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")

        X, _ = self._build_vocabulary(documents, True)
        # Topic_X=[]
        # for doc in X:
        # weight = np.zeros(self.number_of_topics)
        # for topic_id, prob in self.lda_[doc]:
        # weight[topic_id] = prob
        # Topic_X.append(weight)



        return self._fit_doc_topic(X)


        # def most_discussed_topic(self):
        # weight = np.zeros(self.number_of_topics)
        # for doc in self.corpus_tfidf:
        # for col, val in self.lda[doc]:
        # weight[col] += val
        #
        # max_topic = weight.argmax()
        # return max_topic


class LIWC(BaseEstimator):
    def __init__(self, lang):
        self.lang = lang
        self.liwc = self._initialize()


    def _initialize(self):
        return read_topic_tokens(os.path.join('resources', 'liwc', self.lang + ".txt"), 'LIWC')
        # with codecs.open(os.path.join('resources', 'liwc', self.lang + ".txt"), mode='rb') as f:
        # for line in f.readlines():
        # line = line.rstrip('\r\n')
        # topic_words = line.split(',')
        # liwc['LIWC_' + topic_words[0]] = set(topic_words[1:])
        # # print liwc
        # return liwc

    def get_feature_names(self):
        return np.array([topic for topic in sorted(self.liwc)])

    def fit(self, documents, y=None):
        return self


    def transform(self, documents):
        X = []
        for doc in documents:
            # print doc.content.lower()
            counts = np.zeros(len(self.liwc))
            for i, topic in enumerate(self.get_feature_names()):
                document = doc.content.lower()
                counts[i] = np.sum([document.count(" " + word.lower() + " ") for word in self.liwc[topic]])
            X.append(counts)
        # print X
        if not hasattr(self, 'scalar'):
            self.scalar = preprocessing.StandardScaler().fit(X)

        return self.scalar.transform(np.array(X))


class AgeGenderFeature(BaseEstimator):
    def get_feature_names(self):
        return np.array(['Gender_Male', 'Gender_Female', 'AG_18_24', 'AG_25_34', 'AG_35_49', 'AG_50_xx'])


    def fit(self, documents, y=None):
        return self


    def transform(self, documents):
        gender_age_group = [(doc.author.gender, doc.author.age_group) for doc in documents]
        X = []
        for gender, age_group in gender_age_group:
            # print gender,age_group
            weight = np.zeros(len(self.get_feature_names()))
            if gender == 'M':
                weight[0] = 1
            if gender == 'F':
                weight[1] = 1
            if age_group == '18-24':
                weight[2] = 1
            if age_group == '25-34':
                weight[3] = 1
            if age_group == '35-49':
                weight[4] = 1
            if age_group == '50-xx':
                weight[5] = 1
            X.append(weight)

        return np.array(X)


if __name__ == "__main__":
    data = [Document( name='usr1',document=["Fun i'nn is dho the enjoyment of b4 4get for my girlfriend"]),Document(name='user2',document=[
            ' @username! Amazing Week for MACS - my bf dw dsGrapple at the Garden Promo | Video | Flowrestling http://t.co/2PZH5m78 via @username #done	']),Document(name='user2',document=
            ['"The Unwisdom of Crowds?? Why people-powered my child children are overrated" by @username http://t.co/Jf1xnrgat'])]
    data_new = [
        {'gender': 'M', 'age_group': '18-24', 'content': "Fun i'nn is dho the enjoyment of b4 4get for my girlfriend"},
        {'gender': 'F', 'age_group': '25-34', 'content':
            ' @username! Amazing Week for MACS - my bf dw dsGrapple at the Garden Promo | Video | Flowrestling http://t.co/2PZH5m78 via @username #done	'},
        {'gender': 'F', 'age_group': '50-xx', 'content':
            '"The Unwisdom of Crowds?? Why people-powered my child children are overrated" by @username http://t.co/Jf1xnrgat'}]

    # data = ["Is thises"]

    # feature = FamilialFeatures(lang='en')
    # feature.fit(data)
    # print feature.transform(data)
    #
    # print feature.get_feature_names()

    # tw = TwitterFeatures()
    # tw.fit(data)
    # print tw.transform(data)
    # print tw.get_feature_names()

    # feature = CategoricalCharNgramsVectorizer(all=True, ngram_range=(3, 3))
    # feature.fit(data)
    # print feature.transform(data)
    # print feature.get_feature_names()


    # feature = LDA(n_topics=2)
    # feature.fit(data)
    # print feature.transform([Document(document=["Fun i'nn is dho the enjoyment of b4 4get for my girlfriend'"])])
    #
    # print feature.get_feature_names()

    # feature = LIWC(lang='en')
    # feature.fit(data)
    # print feature.transform(data)
    # print feature.transform([Document(document=
    # ["Fun i'nn is dho the enjoyment of b4 4get for my girlfriend you your had other a alot than'"])])
    #
    # print feature.get_feature_names()

    # feature = AgeGenderFeature()
    # feature.fit(data)
    # print feature.transform([
    #     {'gender': 'M', 'age_group': '18-24', 'content': "Fun i'nn is dho the enjoyment of b4 4get for my girlfriend"}])
    #
    # print feature.get_feature_names()