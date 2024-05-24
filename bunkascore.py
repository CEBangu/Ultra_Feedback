import numpy as np
import pandas as pd
from textacy import text_stats, make_spacy_doc
from scipy.stats import zscore

def linear_metric(data:pd.Series) -> tuple: 
    '''This funciton takes in a pandas series of text and returns a DataFrame of the series and each element's
    linear bunka score, as well as a DataFrame of the individual scores for each base metric

    It converts the string into a SpaCy doc, and calculates the zscore of the 
    number of unique words, number of longwords, number of characters, number of syllables per word', 
    number of polysyllabic words, number of words, number of sentences, 'number of monosylabic words,
    entropy of the text, coleman liau score, automated readability score, flesch score, and gunning fog score.
    
    It then combines these scores linearly to make the bunka score, storing them in the linear_df'''
    
    linear_df = pd.DataFrame({'input': data})

    all_df = pd.DataFrame({'input': data})

    for index, text in data.items():
        try:
            response = make_spacy_doc(text, lang='en_core_web_sm')

            all_df.loc[index, 'n_uniquewords'] = text_stats.basics.n_unique_words(response)

            all_df.loc[index, 'n_longwords'] = text_stats.basics.n_long_words(response)

            all_df.loc[index, 'n_chars'] = text_stats.basics.n_chars(response)
        
            all_df.loc[index, 'n_sylsprword'] = text_stats.basics.n_syllables(response)

            all_df.loc[index, 'n_polysylwords'] = text_stats.basics.n_polysyllable_words(response)

            all_df.loc[index, 'n_words'] = text_stats.basics.n_words(response)

            all_df.loc[index, 'n_sents'] = text_stats.basics.n_sents(response)

            all_df.loc[index, 'n_monosylwords'] = text_stats.basics.n_monosyllable_words(response)

            all_df.loc[index, 'entropy'] = text_stats.basics.entropy(response)
        
            all_df.loc[index, 'coleman_liau'] = text_stats.readability.coleman_liau_index(response) 

            all_df.loc[index, 'automated_readability'] = text_stats.readability.automated_readability_index(response)

            all_df.loc[index, 'flesch_kincaid'] = text_stats.readability.flesch_kincaid_grade_level(response) 

            all_df.loc[index, 'gunning_fog'] = text_stats.readability.gunning_fog_index(response) 
        
        except ValueError:
            print(f"SpaCy didn't like the input at index {index}. Try removing it and running again")
    
    linear_df.loc[:, 'linear_bunka_score'] = all_df.iloc[:, 1:].apply(zscore).sum(axis=1)

    return linear_df, all_df