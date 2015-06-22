import collections
from sklearn import preprocessing
from sklearn.cross_validation import ShuffleSplit, KFold, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from features import FamilialFeatures, TwitterFeatures, CategoricalCharNgramsVectorizer, LDA, NGramTfidfVectorizer, LIWC
from model import Document, Author, PersonalityTraits
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import xml.etree.ElementTree as ET
import os
import re
import codecs

import numpy as np

import pickle
from utils import change_range


class Configuration:
    """
    class to control the combination of features
    """
    def __init__(self, kwargs):
        self.language = kwargs.pop('lang', 'en')
        self.ngram = kwargs.pop('ngram', None)
        self.familiar_token = kwargs.pop('familial_token', None)
        self.categorical_char_ngram = kwargs.pop('categorical_char_ngram', None)
        self.liwc = kwargs.pop('liwc', None)
        self.lda = kwargs.pop('lda', None)
        self.twitter = kwargs.pop('twitter', None)


label_extractor_gender = lambda x: x.spit(':::')[1]
label_extractor_age_group = lambda x: x.spit(':::')[2]
label_extractor_age_gender = lambda x: x.split(':::')[1] + "_" + x.split(":::")[2]
# label_extractor_personality = lambda x: [float(x.split(':::')[3]) , float(x.split(":::")[4]),float(x.split(":::")[5]),float(x.split(":::")[6]),float(x.split(":::")[7])]
# label_extractor_traits=lambda  x:x.split(':::')[1] + "_" + x.split(":::")[2]


def label_extractor_personality(x):
    target = ['extroverted', 'stable', 'agreeable', 'conscientious', 'open']
    result = []
    for i, value in enumerate(x.split(":::")):
        if i > 2:
            if float(value) >= 0:
                result.append(target[i - 3])
    return result


class TrainingData:
    def __init__(self, path):
        self.path = path
        self.load_truth()


    # user514:::F:::25-34:::0.0:::0.3:::0.2:::0.2:::0.2

    def load_truth(self):
        truth = {}
        with codecs.open(os.path.join(self.path, 'truth.txt'), 'rb') as f_in:
            for line in f_in.readlines():
                line = line.rstrip('\r\n')
                user_id = line.split(":::")[0]
                truth[user_id] = line
        self.truth = truth


    def load_data(self):
        X, Y = [], []
        for file in os.listdir(self.path):
            if file == 'truth.txt' or file == '.DS_Store':
                continue
            print "loading file -->" + file
            tree = ET.parse(os.path.join(self.path, file))
            root = tree.getroot()
            document = Document(language=root.get('lang'), name=root.get('id'))
            for d in root.findall('document'):
                document.add_document(d.text)
            user, gender, age_group, extroverted, stable, agreeable, conscientious, open = self.truth[
                root.get('id')].split(":::")

            traits = PersonalityTraits(extroverted=float(extroverted), stable=float(stable), agreeable=float(agreeable),
                                       conscientious=float(conscientious), open=float(open))
            usr = Author(gender=gender, age_group=age_group, traits=traits)
            document.author = usr
            X.append(document)
            Y.append(self.truth[root.get('id')])
        print "done loading files"

        self.X = X
        self.Y = Y
        return self


    def transform_label(self, label_extractor, label_encoder=None):
        labels = [label_extractor(label) for label in self.Y]
        if label_encoder:
            return label_encoder.fit_transform(labels)

        return labels


def set_attribute(obj, kwargs):
    for key, value in kwargs.iteritems():
        # print key, value
        setattr(obj, key, value)


class Train:
    def __init__(self, configuration, classifier):
        self.configuration = configuration
        self.classifier = classifier

    def _unify_features(self):
        features = []
        if self.configuration.ngram:
            ngram = NGramTfidfVectorizer()
            set_attribute(ngram, self.configuration.ngram)
            features.append(('ngram', ngram))
            # print ngram.ngram_range
        if self.configuration.familiar_token:
            familial = FamilialFeatures()
            setattr(familial, 'lang', self.configuration.language)
            features.append(('familiar', familial))
        if self.configuration.categorical_char_ngram:
            char_ngram = CategoricalCharNgramsVectorizer()
            set_attribute(char_ngram, self.configuration.categorical_char_ngram)
            features.append(('catcharngram', char_ngram))
        if self.configuration.liwc:
            liwc = LIWC()
            setattr(familial, 'lang', self.configuration.language)
            features.append(('liwc', liwc))
        if self.configuration.lda:
            lda = LDA()
            set_attribute(lda, self.configuration.lda)
            features.append(('lda', lda))
        if self.configuration.twitter:
            twitter = TwitterFeatures()
            features.append(('twitter', twitter))

        all_features = FeatureUnion(features)

        return all_features


    def create_model(self, params=None):
        all_features = self._unify_features()
        pipeline = Pipeline([('all', all_features), ('clf', self.classifier)])
        if params:
            pipeline.set_params(**params)

        return pipeline


    def train(self, X, Y, params=None):
        # print len(X),X
        pipeline = self.create_model(params)
        pipeline.fit(X, Y)
        self.model = pipeline
        return self


    def save(self, file):
        pickle.dump(self.model, open(file, "w"))


label_extractor_gender_age = lambda x: {'gender': x.split('_')[0], 'age_group': x.split('_')[1]}
label_extractor_personality_values = lambda target, x: {label: x[i] for i, label in enumerate(target)}

Model = collections.namedtuple('Model', 'name clf label_encoder label_extractor')


class Test:
    def __init__(self, model, path):
        self.model = model
        self.path = path

        #self.load_truth()


    def load_truth(self):
        truth = {}
        with codecs.open(os.path.join(self.path, 'truth.txt'), 'rb') as f_in:
            for line in f_in.readlines():
                line = line.rstrip('\r\n')
                user_id = line.split(":::")[0]
                truth[user_id] = line
        self.truth = truth


    def run(self):
        result = {}
        x, y, y_actual = [], [], []
        for file in os.listdir(self.path):
            if file == 'truth.txt' or file == '.DS_Store':
                continue
            tree = ET.parse(os.path.join(self.path, file))
            root = tree.getroot()
            document = Document(language=root.get('lang'), name=root.get('id'))

            for d in root.findall('document'):
                document.add_document(d.text)
            x_test = [document]  # vector

            temp_result = {}
            for predictor in self.model:
                # print predictor

                if predictor.name == 'age_gender':
                    prediction = predictor.clf.predict(x_test)  # predict
                    temp_result.update(
                        predictor.label_extractor(list(predictor.label_encoder.inverse_transform(prediction))[0]))
                    document.author.gender = temp_result['gender']
                    document.author.age_group = temp_result['age_group']
                if predictor.name == 'personality':
                    target = predictor.label_encoder.classes_
                    prediction = list(predictor.clf.predict_proba(x_test))[0]
                    prediction = [change_range(p, 1.0, 0.0, 0.5, -0.5) for p in prediction]
                    temp_result.update(predictor.label_extractor(target, prediction))



            document.author.personality_traits.extroverted = temp_result['extroverted']
            document.author.personality_traits.agreeable = temp_result['agreeable']
            document.author.personality_traits.conscientious = temp_result['conscientious']
            document.author.personality_traits.stable = temp_result['stable']
            document.author.personality_traits.open = temp_result['open']

            result[os.path.splitext(file)[0]] = document
            # y.extend(prediction)
            # print y
            x.append(os.path.splitext(file)[0])
            # y_actual.append(predictor.label_extractor(self.truth[root.get('id')]))
        self.x_test = x_test
        # self.y_prediction = y
        # self.y_actual = self.label_encoder.transform(y_actual)
        self.result = result


    def save(self, output):
        for user, result in self.result.iteritems():
            with codecs.open(os.path.join(output, user + ".xml"), 'wb') as out:
                out.write('<author id="' + user + '"\n\t' +
                          'type="' + 'twitter' + '"\n\t' +
                          'lang="' + result.lang + '"\n\t' +
                          'age_group="' + result.author.age_group + '"\n\t' +
                          'gender="' + result.author.gender + '"\n\t' +
                          'extroverted="' + result.author.extroverted + '"\n\t' +
                          'stable="' + result.author.stable + '"\n\t' +
                          'agreeable="' + result.author.agreeable + '"\n\t' +
                          'conscientious="' + result.author.conscientious + '"\n\t' +
                          'open="' + result.author.open + '"\n\t' +
                          '/>')


    def report(self):
        print(classification_report(self.y_actual, self.y_prediction, target_names=self.label_encoder.classes_))
        print "\n======================================================"
        print "\n            Confusion Matrix"
        print confusion_matrix(self.y_actual, self.y_prediction)
        print "\n======================================================"
        print "\n Accuracy ->" + str(accuracy_score(self.y_actual, self.y_prediction))
        print "\n======================================================"


class CrossFoldValidation:
    def __init__(self, n_fold, X, Y, clf, **kwargs):
        self.n_fold = n_fold
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.clf = clf
        self.n_iter = kwargs.get('n_iter', 10)
        self.test_size = kwargs.get('test_size', 0.3)

    def run_shuffle_split(self):
        # print self.Y

        cv = ShuffleSplit(n=len(self.X), n_iter=self.n_iter, test_size=self.test_size, random_state=0)
        scores = []
        counter = 0
        for train, test in cv:
            # print train,test
            counter += 1
            X_train, y_train = self.X[train], self.Y[train]
            X_test, y_test = self.X[test], self.Y[test]

            self.clf.fit(X_train, y_train)
            # train_score = self.clf.score(X_train, y_train) #accuracy
            test_score = self.clf.score(X_test, y_test)  # accuracy

            scores.append(test_score)
            print 'fold' + str(counter) + "  done ..."

        summary = (np.mean(scores), np.std(scores))
        print "Accuracy -> %.3f\t Standard Deviation-> %.3f\t" % summary


    def run_k_fold(self):
        cv = KFold(n=len(self.X), n_folds=self.n_iter, indices=True)
        scores = []
        counter = 0
        for train, test in cv:
            # print train,test
            counter += 1
            X_train, y_train = self.X[train], self.Y[train]
            X_test, y_test = self.X[test], self.Y[test]

            self.clf.fit(X_train, y_train)
            # train_score = self.clf.score(X_train, y_train) #accuracy
            test_score = self.clf.score(X_test, y_test)  # accuracy

            scores.append(test_score)
            print 'fold' + str(counter) + "  done ..."
        summary = (np.mean(scores), np.std(scores))
        print "Accuracy -> %.3f\t Standard Deviation-> %.3f\t" % summary


    def run_stratified_kfold(self):
        cv = StratifiedKFold(self.Y, n_folds=self.n_iter, indices=True)
        scores = []
        counter = 0
        for train, test in cv:
            # print train,test
            counter += 1
            X_train, y_train = self.X[train], self.Y[train]
            X_test, y_test = self.X[test], self.Y[test]

            self.clf.fit(X_train, y_train)
            # train_score = self.clf.score(X_train, y_train) #accuracy
            test_score = self.clf.score(X_test, y_test)  # accuracy

            scores.append(test_score)
            print 'fold' + str(counter) + "  done ..."
        summary = (np.mean(scores), np.std(scores))
        print "Accuracy -> %.3f\t Standard Deviation-> %.3f\t" % summary


if __name__ == "__main__":
    train_data_path = "/Users/suraj/PycharmProjects/authorprofile/resources/en/test"
    test_data_path = ''
    le = preprocessing.LabelEncoder()
    train_data = TrainingData(path=train_data_path, label_encoder=le).load_data()
