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
    df = pd.read_csv(path)
    data = df.filter(items=['Device Name', 'Start Time', 'Transcript'])
    data['Start Time'] = pd.to_datetime(data['Start Time'])
    data = data.set_index('Start Time')    
    combined_data = data.groupby(pd.Grouper(freq='30s')).agg({'Device Name': 'first', 'Transcript': lambda x: ' '.join(x)})
    processed_data = pd.DataFrame(columns=['Start Time', 'Transcript', 'Transcript Length'])
    i = 0
    current_transcript = ''
    current_length = 0
    
    for index, row in combined_data.iterrows():
        name = row['Device Name']
        transcript = row['Transcript']
        
        if current_length + len(f"{name} says '{transcript}'. ") > max_chars:
            processed_data.loc[i] = [index, current_transcript.strip(), current_length]
            i += 1
            current_transcript = ''
            current_length = 0
        
        current_transcript += f"{name} says '{transcript}'. "
        current_length += len(f"{name} says '{transcript}'. ")
    
    processed_data.loc[i] = [index, current_transcript.strip(), current_length]
    #processed_data = processed_data.set_index('Start Time')
    return processed_data

def raw_summary(text):
    tokens = simple_preprocess(text)
    bow = gensim.corpora.Dictionary([tokens])
    summary = summarize(text,0.05)
    return summary

def final_summary(text):
    detailed_prompt = '''Summarize the information and topics discussed in the conversation
                Always following the format: 
                Speaker involves: Name of the speakers
                Overall Summary: One sentence covering the major topics of the conversation
                Details and Decisions: Using bullet points to summary 3-5 topics appears in the conversation
                
                Noted that you should void any repetition along the way.
                
                The raw text needed to summarized:
            '''
    prompt = detailed_prompt + text
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      max_tokens=500,
      n=1,
      stop=None,
      temperature=0.5,
    )
    summary = response.choices[0].text.strip()
    return summary

def summary_generator(processed_df):
    output = pd.DataFrame(data = {'Start Time':[],'Gensim Summary':[],'GPT Summary':[],'Word Count':[],'Gensim Word Count':[],'GPT Word Count':[]})
    output['Start Time'] = list(processed_df['Start Time'])
    gensim_summary = []
    gpt_summary= []
    gensim_wordcount = []
    gpt_wordcount = []
    wordcount = []
    for index, row in processed_df.iterrows():
        print(f"now processing row ",index)
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
#processed_df.to_csv('token_cut.csv', index=False)
df = summary_generator(processed_df)
df.to_csv('Talk_format_bytoken.csv', index=False)

end_time = time.time()
run_time = end_time - start_time
print(f"Total running time: {run_time} seconds")