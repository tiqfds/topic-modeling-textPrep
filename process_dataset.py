from nltk.corpus import stopwords

from settings.common import word_tf_df
from preprocessing_pipeline.NextGen import NextGen
from preprocessing_pipeline import (Preprocess, RemovePunctuation, Capitalization, RemoveStopWords,
                                    RemoveShortWords, TwitterCleaner, RemoveUrls)

from pprint import pprint
import nltk 
nltk.download()


def load_flat_dataset(path):
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(line.strip().split(' '))
    return dataset

def load_dataset_with_dates(path):
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(line.strip().split('\t')[0].split(' '))
        return dataset
    except FileNotFoundError:
        print('The path provided for your dataset does not exist: {}'.format(path))
        import sys
        sys.exit()

import csv


if __name__ == '__main__':

    dataset_test = load_flat_dataset('data/full_raw_data_EDA.csv')
    pprint(dataset_test[:5])

    dataset_names = ['full_raw_data_EDA']
    for j in range(0, len(dataset_names)):
        ds = dataset_names[j]

        stopwords_list = stopwords.words('english')
        stopwords_list.append(['rt', 'amp'])

        pipeline = Preprocess()
        rp = RemovePunctuation(keep_hashtags=False)
        ru = RemoveUrls()
        cap = Capitalization()
        short_words = RemoveShortWords()
        tc = TwitterCleaner()
        rsw = RemoveStopWords(extra_sw=stopwords_list)

        pipeline.document_methods = [(tc.remove_deleted_tweets, str(tc),),
                                     (tc.remove_users, str(tc),),
                                     (ru.remove_urls, str(ru),),
                                     (rp.remove_punctuation, str(rp),),
                                     (cap.lowercase, str(cap),),
                                     (tc.remove_rt, str(tc),),
                                     (rsw.remove_stopwords, str(rsw),),
                                     (short_words.remove_short_words, str(short_words),)
                                     ]

        ng = NextGen()
        path = 'data/{}.csv'.format(ds)
        dataset = load_dataset_with_dates(path)
        processed_dataset = ng.full_preprocess(dataset, pipeline, ngram_min_freq=10, extra_bigrams=None)

        with open('data/{}_lightweight.csv'.format(ds), 'w') as f:
            for i in range(0, len(processed_dataset)):
                doc = processed_dataset[i]
                f.write('{}\n'.format(' '.join(doc)))

        freq = {}
        freq = word_tf_df(freq, processed_dataset)   
        processed_dataset = ng.filter_by_tfidf(dataset=processed_dataset, freq=freq, threshold=0.25)

        # with open('data/{}_lightweight_tdidf.csv'.format(ds), 'w') as f:
        #     for i in range(0, len(processed_dataset)):
        #         doc = processed_dataset[i]
        #         f.write('{}\n'.format(' '.join(doc)))
