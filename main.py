import openai
import streamlit as st
import pandas as pd
import os 
from io import StringIO
from streamlit_chat import message
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
# pip install streamlit-chat  
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://dbhackathonai6-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "c59d9e19c07942889de26abe78a02e09"


hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """


#openai.api_key = st.secrets['api_key']

st.set_page_config(page_title="Ask Anything App")
st.markdown(hide_default_format, unsafe_allow_html=True)
#Creating the chatbot interface
st.title("Welcome in world of AI....")


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm Your's buddy, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi Start asking..']

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()


    
# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    st.button('Ask me.')
   # input_text = st.file_uploader("Choose a CSV file", type='csv')
    #query = 'summarise the document in bullet point'
    return input_text
    


## Applying the user input box
with input_container:
    user_input = get_text()


def generate_response(prompt):
    documents = []
    # os.chdir(r"C:\Users\Tejas\Downloads\openai-lab\openai-lab\documents")
    docs = r"C:\Users\Kartikeya Mishra\Downloads\Gaurav\Hackathon\ChatBotApi\Documents\\"
    for file in os.listdir(docs):
        if file.endswith(".pdf"):
            pdf_path = r"C:\Users\Kartikeya Mishra\Downloads\Gaurav\Hackathon\ChatBotApi\Documents\\" + file
            loader = PyPDFDirectoryLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = r"C:\Users\Kartikeya Mishra\Downloads\Gaurav\Hackathon\ChatBotApi\Documents\\" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = r"C:\Users\Kartikeya Mishra\Downloads\Gaurav\Hackathon\ChatBotApi\Documents\\" + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())

    # Index that wraps above steps
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)

    llm = AzureOpenAI(
        deployment_name="gpt-35-turbo",
        model_name="gpt-35-turbo",
    )

    chain = load_qa_chain(llm,chain_type="stuff")
    #dta = chain({"input_documents": documents, "question": prompt},return_only_outputs=True)
    #query = 'summarise the document in bullet point'
 
    response = chain.run(input_documents=documents, question=prompt,return_only_outputs=True)
    index = response.index('Question') if 'Question' in response else -1
    #print(response)
    #print(response[0:response.index('Question')])
    # if(index!=-1):
    #     return response[0:response.index('Question')]
    # else:
    #     return response

    return response[0:response.index('\n')]
    #return response


with st.sidebar:
    add_radio = st.radio(
        "AskGPT...",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )

    
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
       

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))


