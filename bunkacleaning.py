import pandas as pd
import numpy as np
from langdetect import detect, DetectorFactory
from textacy import text_stats, make_spacy_doc


def bad_idx_finder(DataFrame):
    return np.where(DataFrame.iloc[:, 3] < DataFrame.iloc[:, 4])[0]

def data_switcher(badidxs: list, DataFrame): 
    storage = {'metric_accept': [], 'metric_accept_score': [], 'metric_reject': [], 'metric_reject_score': []}

    #populate the storage dict
    for idx in badidxs:
        storage['metric_accept'].append(DataFrame.iloc[idx, 2]) 
        storage['metric_accept_score'].append(DataFrame.iloc[idx, 4])

        storage['metric_reject'].append(DataFrame.iloc[idx, 1])
        storage['metric_reject_score'].append(DataFrame.iloc[idx, 3])

    mask = DataFrame.index.isin(badidxs)

    DataFrame.loc[mask, DataFrame.columns[1]] = storage['metric_accept']
    DataFrame.loc[mask, DataFrame.columns[3]] = storage['metric_accept_score']

    DataFrame.loc[mask, DataFrame.columns[2]] = storage['metric_reject']
    DataFrame.loc[mask, DataFrame.columns[4]] = storage['metric_reject_score']

    DataFrame['absolute_difference'] = abs(DataFrame.iloc[:, -1])

    return DataFrame


def lang_cull(DataFrame, response_column: str, prompt_column: str) -> tuple:
    
    DetectorFactory.seed = 0

    df = DataFrame.copy()

    non_english_prompts = []
    non_parasable_prompts = []

    for _, row in df.iterrows():
        try:
            if detect(row[response_column]) != 'en':
                non_english_prompts.append(row[prompt_column])
        except:
            non_parasable_prompts.append(row[prompt_column])

    unusable_prompts = non_english_prompts + non_parasable_prompts
    clean_df = df[~df[prompt_column].isin(unusable_prompts)]

    return clean_df, unusable_prompts

def get_stats(data:pd.Series): 
    '''This function takes in a pandas series of text and returns a DataFrame of the series and each element's
    linear bunka score, as well as a DataFrame of the individual scores for each base metric

    It converts the string into a SpaCy doc, and calculates the score of the 
    number of unique words, number of longwords, number of characters, number of syllables per word', 
    number of polysyllabic words, number of words, number of sentences, number of monosylabic words,
    entropy of the text, coleman liau score, automated readability score, flesch score, and gunning fog score.
    
    It returns a dataframe with these scores next to the data, which you can merge with the original on a shared column'''
    
    df = pd.DataFrame({'input': data})

    for index, text in data.items():
        try:
            response = make_spacy_doc(text, lang='en_core_web_sm')

            df.loc[index, 'n_uniquewords'] = text_stats.basics.n_unique_words(response)

            df.loc[index, 'n_longwords'] = text_stats.basics.n_long_words(response)

            df.loc[index, 'n_chars'] = text_stats.basics.n_chars(response)
        
            df.loc[index, 'n_sylsprword'] = text_stats.basics.n_syllables(response)

            df.loc[index, 'n_polysylwords'] = text_stats.basics.n_polysyllable_words(response)

            df.loc[index, 'n_words'] = text_stats.basics.n_words(response)

            df.loc[index, 'n_sents'] = text_stats.basics.n_sents(response)

            df.loc[index, 'n_monosylwords'] = text_stats.basics.n_monosyllable_words(response)

            df.loc[index, 'entropy'] = text_stats.basics.entropy(response)
        
            df.loc[index, 'coleman_liau'] = text_stats.readability.coleman_liau_index(response) 

            df.loc[index, 'automated_readability'] = text_stats.readability.automated_readability_index(response)

            df.loc[index, 'flesch_score'] = text_stats.readability.flesch_reading_ease(response) 

            df.loc[index, 'gunning_fog'] = text_stats.readability.gunning_fog_index(response) 
        
        except ValueError:
            print(f"SpaCy didn't like the input at index {index}. Try removing it and running again")

    return df