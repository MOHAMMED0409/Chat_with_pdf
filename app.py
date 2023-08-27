import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # extract page
        for page in pdf_reader.pages:
            text += page.extract_text()  # extract text from each page
    return text


def get_text_chucks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chucks = text_splitter.split_text(text)
    return chucks


def get_vectorstore(text_chucks):
    # embeddings = OpenAIEmbeddings() pay to use
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chucks, embedding=embeddings)
    return vectorstore


def get_conversation(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDF's", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.header("Chat with PDF's :books:")

    user_question = st.text_input("Ask Questions ?")

    if user_question:
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(user_template.replace("{{MSG}}", "Hello Bot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Champions"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDF's")
        pdf_docs = st.file_uploader(
            "Upload the PDF's Here and click on'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get text chucks
                text_chucks = get_text_chucks(raw_text)
                # st.write(text_chucks)

                # create vector storage
                vectorstore = get_vectorstore(text_chucks)

                # create conversation
                st.session_state.conversation = get_conversation(vectorstore)


if __name__ == "__main__":
    main()
