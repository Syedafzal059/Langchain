from streamlit_chat import message 
import streamlit as st
from EduBot import *
import langchain


#langchain.globals.set_verbose()
  


def create_edubot():
    edubotcreator = EduBotCreator()
    edubot = edubotcreator.create_edubot()
    print(f" the type is{type(edubot)}")
    return edubot


edubot = create_edubot()



def infer_edubot(prompt):
    model_out = edubot(prompt)
    answer = model_out['result']
    return answer

def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str[i]+"_user")
        message(history["assistant"][i], key=str(i))

    


def main():

    st.title("Edubot: Your Smart Education Sidekick")
    st.subheader("A bot created using Langchain to on cpu making your learning process easy")
    user_input = st.text_input("Enter Your Query")



    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["I am ready to help you"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey There!"]


    if st.button("Answer"):
        print(user_input)
        answer = infer_edubot({'query': user_input})
        st.session_state["user"].append(user_input)
        st.session_state["assistant"].append(answer)

        if st.session_state["assistant"]:
            display_conversation(st.session_state)



if __name__ == "__main__":
    main()