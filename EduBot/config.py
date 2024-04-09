DATA_DIR_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDER = "thenlper/gte-large"
VECTOR_DB_PATH = "faiss/education"
DEVICE = "cpu"


PROMPT_TEMPLATE  = '''
With the information provided try to answer the question.
If yoy cant answer the question based on the information provoided either say you cant find an answer or unable to find the answer
So try to understand in depth about the context and answer only based on the information provided.Dont generate random output


Context: {context}
Question: {question}

Do provide only helpfull answers
Helpful answer:
'''
INP_VARS = "['context', 'question]"
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k':2}
MODEL_CKPT = "model/llama-2-7b-chat.ggmlv3.q2_K.bin"
MODEL_TYPE = "llama"
MAX_NEW_TOKENS = 512
TEMPERATURE  = 0.9