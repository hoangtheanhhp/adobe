import gensim
import nltk
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis
from tqdm import tqdm_notebook

stop_words = nltk.download('stopwords')

df = pd.read_table('QQP/dev.tsv')
df = df.dropna(how="any").reset_index(drop=True)
a = 0
for i in range(a, a + 10):
    print(df.question1[i])
    print(df.question2[i])
    print()


def wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    stop_words = stopwords.words('english')
    q1 = [w for w in q1 if w not in stop_words]
    q2 = [w for w in q2 if w not in stop_words]
    return model.wmdistance(q1, q2)


def norm_wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    stop_words = stopwords.words('english')
    q1 = [w for w in q1 if w not in stop_words]
    q2 = [w for w in q2 if w not in stop_words]
    return norm_model.wmdistance(q1, q2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


df.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)

df['len_q1'] = df.question1.apply(lambda x: len(str(x)))
df['len_q2'] = df.question2.apply(lambda x: len(str(x)))
df['diff_len'] = df.len_q1 - df.len_q2
df['len_char_q1'] = df.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
df['len_char_q2'] = df.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
df['len_word_q1'] = df.question1.apply(lambda x: len(str(x).split()))
df['len_word_q2'] = df.question2.apply(lambda x: len(str(x).split()))
df['common_words'] = df.apply(
    lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),
    axis=1)
df['fuzz_ratio'] = df.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_partial_token_set_ratio'] = df.apply(
    lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_partial_token_sort_ratio'] = df.apply(
    lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])),
                                       axis=1)
df.head(2)
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
df['wmd'] = df.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)

df.head(2)

norm_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model.init_sims(replace=True)
df['norm_wmd'] = df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)

df.head(2)

question1_vectors = np.zeros((df.shape[0], 300))

for i, q in enumerate(tqdm_notebook(df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors = np.zeros((df.shape[0], 300))
for i, q in enumerate(tqdm_notebook(df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

df['cosine_distance'] = [cosine(x, y) for (x, y) in
                         zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['cityblock_distance'] = [cityblock(x, y) for (x, y) in
                            zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['jaccard_distance'] = [jaccard(x, y) for (x, y) in
                          zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['canberra_distance'] = [canberra(x, y) for (x, y) in
                           zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['euclidean_distance'] = [euclidean(x, y) for (x, y) in
                            zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in
                            zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in
                             zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]
df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

df['is_duplicate'].value_counts()

df.isnull().sum()

df.drop(['question1', 'question2'], axis=1, inplace=True)
df = df[pd.notnull(df['cosine_distance'])]
df = df[pd.notnull(df['jaccard_distance'])]

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X = df.loc[:, df.columns != 'is_duplicate']
y = df.loc[:, df.columns == 'is_duplicate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

import xgboost as xgb

model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4,
                          objective='binary:logistic', eta=0.3, silent=1, subsample=0.8).fit(X_train,
                                                                                             y_train.values.ravel())
prediction = model.predict(X_test)
cm = confusion_matrix(y_test, prediction)
print(cm)
print('Accuracy', accuracy_score(y_test, prediction))
print(classification_report(y_test, prediction))
