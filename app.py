import streamlit as st
import pandas as pd
import base64
import csv
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import docx
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

# Initialize chat model
chat_llm = ChatOpenAI(temperature=0.0)

# Function to convert dictionary to CSV
def dict_to_csv(data, filename, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not append:
            writer.writeheader()
        writer.writerow(data)

# Load paraphrasing model
# model_name = 'tuner007/pegasus_paraphrase'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# Function to generate paraphrased text
def get_response(input_text, num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=60, num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def split_into_batches(paragraph, batch_size):
    sentences = paragraph.split(". ")  # Assuming sentences are separated by ". "
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    return batches


# Function to process the CSV and perform policy generation
def result(df):
    Policy_schema = ResponseSchema(name="Policy", description="Policy Statement")
    response_schemas = [Policy_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    title_template = """ \ You are an AI Governance bot. Just Execute the set of steps one by one.
                Convert "{topic}" into policy statements in 10 words.
                {format_instructions}
                """ 
    prompt = ChatPromptTemplate.from_template(template=title_template)
    df2 = df["Combined"]
    
    for i in range(len(df2)):
        messages = prompt.format_messages(topic=df2[i], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = output_parser.parse(response.content)
        data = response_as_dict
        dict_to_csv(data, 'policy.csv', append=True)
    
    result = pd.read_csv("policy.csv", names=['Policy'])
    final = pd.concat([df, result], axis=1)
    final.to_csv("result1.csv")
    st.dataframe(final)

# Function to perform paraphrasing
# def result2(df):
#     df4 = df['Policy']
#     paraphrase = []
#     for i in df['Policy']:
#         a = get_response(i, 1)
#         paraphrase.append(a)

#     df['paraphrased_text'] = paraphrase
#     df.to_csv('result2.csv', index=False)
#     st.markdown("Paraphrased Successfully")

# Function to perform topic generation and summarization
def result3(df):
    
    
    df5 = df['Policy'].str.replace('[\[\]]', '', regex=True)
    df['Policy'] = df['Policy'].str.replace('[\[\]]', '', regex=True)

    # Combine all rows into a single variable
    combined_text = ' '.join(df['Policy'].astype(str))
    
    title_template = """ \ You are an AI Governance bot. 
                summarize the "{topic}" in one or two instructional policy pointers under each topic.
                """ 
    # for 10 batch classify and group the statements into key topic
    prompt = ChatPromptTemplate.from_template(template=title_template)
    
    
    batch_size =5
    batches = split_into_batches(combined_text, batch_size)

    
    if os.path.exists('test.doc'):
        doc = docx.Document('test.doc')
    else:
        doc = docx.Document()
    
    for batch in batches:
        paragraph = ". ".join(batch)
        messages = prompt.format_messages(topic=paragraph)
        response = chat_llm(messages)
        content = str(response.content)  # Assuming response.content is a string
        doc.add_paragraph(content)
    
    doc.save('test.doc')
    # doc = docx.Document() 
    # messages = prompt.format_messages(topic=combined_text)
    # response = chat_llm(messages)
    # content = str(response.content)  # Assuming response.content is a string
    # doc.add_paragraph(content)
    # doc.save('test.doc')

    with open('test.doc', 'rb') as f:
        doc_data = f.read()
    b64 = base64.b64encode(doc_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="result.doc">Download Result</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    
# Function to concatenate columns and download the modified CSV file
def process_csv(df):
    # df = pd.read_csv(file)
    df['Combined'] = df['OP Title'].astype(str) + df['OP Description'].astype(str)
    modified_csv = df.to_csv(index=False)
    
    # Download the modified CSV file
    b64 = base64.b64encode(modified_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="modified.csv">Download modified CSV file</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Show a preview of the modified data
    st.dataframe(df)
    df.to_csv('policy_md1.csv')

# Streamlit app
def main():
    st.title("Policy Prodago")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    
    if uploaded_file is not None:
        st.write("File uploaded successfully!")
        df = pd.read_csv(uploaded_file)
        
        st.subheader("CSV File Preview")
        st.dataframe(df)
        # Process button
        if st.button("Process"):
            process_csv(df)
        
        if st.button("Policy Generation"):
            result(pd.read_csv("policy_md1.csv"))
        
        # if st.button("Paraphraser"):
        #     result2(pd.read_csv("result1.csv"))
            
        if st.button("Topic Generation"):
            result3(pd.read_csv('result1.csv'))

if __name__ == "__main__":
    main()
