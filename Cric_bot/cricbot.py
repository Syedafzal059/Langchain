from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
import tempfile



def load_llm():
    llm = CTransformers(
        model = "",
        model_type = "llama",
        max_new_tokens = 512,
        temperature= 0.9 
    )
    return llm



#Made the streamlit app
st.title("Cricbot- Chat with Cricket CSV Data")
csv_data = st.sidebar.file_uploader("Upload your Data", type="csv")

if csv_data:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(csv_data.getvalue())
        tmp_file_path = tmp_file.name
    

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter':','})
    data = loader.load()
    st.json(data)
    embeddings = HuggingFaceHubEmbeddings(model='thenlper/gte-large')#, model_kwargs={'device':'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local('faiss/cricket')
    llm = load_llm()

    prompt_temp = '''
    With the information provided try to answer the question.
    If you cant answer the question based on the information either say you cant find the answer or unable to find.
    This is related to the cricket domain. so try to understand in depth about the context and answer only based on the data provided
    Context:{context}
    Question:{question}
    Do provide only correct answers

    Correct answer:
    '''
    custom_prompt_temp = PromptTemplate(template=prompt_temp,
                                        input_variables=['context', 'question'])
    retrieval_qa_chain = RetrievalQA.from_chain_type(

        llm = llm,
        retriver = db.as_retriever(search_kwargs={'k': 1}),
        chain_type = "stuff",
        return_source_documents = True,
        chain_type_kwargs= {"prompt":custom_prompt_temp}
    )


    def cricbot(query):
        answer = retrieval_qa_chain({"query":query})
        return answer["result"]
    
    if 'user' not in st.session_state:
        st.session_state['user'] = ["Hey there"]

    if 'assistant' not in st.session_state:
        st.session_state['assistant']= ["Hello I am Cricbot and I am ready to help "]

    container = st.container()
    with container:
        with st.form(key = 'cricket_form', clear_on_submit=True):
            user_input = st.text_input("",placeholder="Talk to your csv data here", key= "input")
            submit = st.form_submit_button(label='Answer')
        if submit:
            output = cricbot(user_input)
            st.session_state['user'].append(user_input)
            st.session_state['assistant'].append(output)

    if st.session_state['assistant']:
        for i in range(len(st.session_state['assistant'])):
            message(st.session_state["user"][i], is_user= True, key=str(i)+'_user') 
            message(st.session_state["assistant"][i], key= str(i))


