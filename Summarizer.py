## Inputs: API Key, CSV file, Preference
## Outputs: CSV file
import pandas as pd
import time
from typing import List
import gensim
from gensim.summarization import summarize
from gensim.utils import simple_preprocess

import openai
openai.api_key = "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"

start_time = time.time()



def processor(path, max_chars=50000):
    df = pd.read_csv(path) # reads in a CSV file from the given path
    data = df.filter(items=['Device Name', 'Start Time', 'Transcript']) # filters out unnecessary columns
    data['Start Time'] = pd.to_datetime(data['Start Time']) # converts 'Start Time' column to datetime format
    data = data.set_index('Start Time') # sets the index to the 'Start Time' column
    
    # groups the data by 30-second intervals and concatenates all transcripts within each interval
    combined_data = data.groupby(pd.Grouper(freq='30s')).agg({'Device Name': 'first', 'Transcript': lambda x: ' '.join(x)})
    # creates an empty dataframe to store the processed data
    processed_data = pd.DataFrame(columns=['Start Time', 'Transcript', 'Transcript Length'])
    
    i = 0
    current_transcript = ''
    current_length = 0
    # iterates through each row of the combined data and splits the concatenated transcripts into chunks that don't exceed the maximum number of characters
    for index, row in combined_data.iterrows():
        name = row['Device Name']
        transcript = row['Transcript']
        
        if current_length + len(transcript) > max_chars:
            # creates a new row in the processed dataframe with the current chunk of transcripts and its length
            processed_data.loc[i] = [index, current_transcript.strip(), current_length]
            i += 1
            current_transcript = ''
            current_length = 0
        
        current_transcript += transcript
        current_length += len(transcript)
        
    # creates a new row in the processed dataframe with the remaining chunk of transcripts and its length
    processed_data.loc[i] = [index, current_transcript.strip(), current_length]
    return processed_data # returns the processed data in a dataframe format.

def raw_summary(text):
    tokens = simple_preprocess(text) # tokenizes the input text using simple_preprocess
    bow = gensim.corpora.Dictionary([tokens]) # creates a Bag-of-Words representation of the tokens using gensim.corpora.Dictionary
    summary = summarize(text,0.05) # generates a summary of the text using the summarization function with a ratio of 0.05
    return summary

def final_summary(text):
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

def summary_generator(processed_df):
    output = pd.DataFrame(data = {'Start Time':[],'Gensim Summary':[],'GPT Summary':[],'Word Count':[],'Gensim Word Count':[],'GPT Word Count':[]})
    output['Start Time'] = list(processed_df['Start Time'])
    gensim_summary = [] # Stores the summary generated by gensim
    gpt_summary= [] # Stores the summary generated by GPT-3.5
    gensim_wordcount = [] # Stores the number of tokens in the gensim summary
    gpt_wordcount = [] # Stores the number of tokens in the GPT summary
    wordcount = [] # Stores the number of tokens in the original text
    
    for index, row in processed_df.iterrows():
        print(f"now processing row {index}")
        text = processed_df['Transcript'][index]
        text_length = processed_df['Transcript Length'][index]
        summary = raw_summary(text)
        output_summary = final_summary(summary)
        gensim_summary.append(summary)
        gpt_summary.append(output_summary)
        gensim_wordcount.append(len(summary))
        gpt_wordcount.append(len(output_summary))
        wordcount.append(text_length)
    output['Gensim Summary'] = gensim_summary
    output['GPT Summary'] = gpt_summary
    output['Gensim Word Count'] = gensim_wordcount
    output['GPT Word Count'] = gpt_wordcount
    output['Word Count'] = wordcount
    return output

processed_df = processor('talk.csv')
print(processed_df['Start Time'])
df = summary_generator(processed_df)
df.to_csv('Output_talk_withoutName.csv', index=False)

'''
end_time = time.time()
run_time = end_time - start_time
print(f"Total running time: {run_time} seconds")
'''