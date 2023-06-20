import pandas as pd
import time
from typing import List
import gensim
from gensim.summarization import summarize
from gensim.utils import simple_preprocess
import os

import openai
openai.api_key = os.env['OPENAI_API_KEY']

def combine_string(*args):
    combined_string = ""
    max_length = 50000
    for string in args:
        if len(combined_string) + len(string) <= max_length:
            combined_string += string
        else:
            break
    return combined_string

def gensim_summary(text):
    tokens = simple_preprocess(text) # tokenizes the input text using simple_preprocess
    bow = gensim.corpora.Dictionary([tokens]) # creates a Bag-of-Words representation of the tokens using gensim.corpora.Dictionary
    summary = summarize(text,0.05) # generates a summary of the text using the summarization function with a ratio of 0.05
    return summary

def gpt_summary(text):
    detailed_prompt = '''Summarize the information and topics discussed in the conversation
                Always following the format: 
                Overall Summary: One sentence covering the major topics of the conversation
                Details and Decisions: Using bullet points to summary 3-5 topics appears in the conversation
                
                Noted that you should void any repetition along the way.
                
                The raw text needed to summarized:
            '''
    prompt = detailed_prompt + text
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=500,  # sets the maximum number of tokens for the generated summary
      n=1,
      stop=None,
      temperature=0.25, # sets the temperature for generating diverse responses
    )
    summary = response.choices[0].text.strip() # extracts the generated summary from the API response
    return summary

def summary_generator(combined_string):
    raw_summary = gensim_summary(combined_string) 
    final_summary = gpt_summary(raw_summary)
    return final_summary

'''
def test(inputpath, outputpath):
    df = pd.read_csv(inputpath)
    result = ''
    for index, row in df.iterrows():
        string = row['Transcript']
        result = combine_string(result, string)
    summary = summary_generator(result)
    with open(outputpath, 'w') as file:
        file.write(summary)
    print(len(result))
    print(len(summary))
        
test('talk.csv','test_1.txt')
'''
