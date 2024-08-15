import telebot
from dotenv import load_dotenv
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Установка переменных среды для API
def initialize_environment():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBFGavIgm697DrT3iUnSrhrBlJmA1_BuCY"

# Чтение содержимого PDF файла
def read_pdf_content(file_path):
    with open(file_path, "rb") as pdf_file:
        reader = PdfReader(pdf_file)
        return "".join(page.extract_text() for page in reader.pages)

# Разбиение текста на сегменты
def partition_text(text, chunk_size=10000, chunk_overlap=1000):
    chunker = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_text(text)

# Создание векторного индекса
def create_vector_index(segments, model_name="models/embedding-001", index_name="faiss_index"):
    embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
    vector_index = FAISS.from_texts(segments, embedding=embedding_model)
    vector_index.save_local(index_name)

# Конфигурация диалоговой цепочки
def configure_chat_chain():
    chain_prompt = """
    В данном файле содержатся различные вопросы и ответы о компьютерных играх. Дайте максимально развернутый ответ на вопрос, используя предоставленный контекст \n\n
    Контекст:\n {context}\n
    Вопрос:\n {question}\n
    Ответ:
    """
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, max_length=10000)
    formatted_prompt = PromptTemplate(template=chain_prompt, input_variables=["context", "question"])
    return load_qa_chain(chat_model, chain_type="stuff", prompt=formatted_prompt)

# Загрузка векторного индекса и обработка запроса пользователя
def handle_user_query(user_input, index_name="faiss_index", model_name="models/embedding-001"):
    embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
    loaded_index = FAISS.load_local(index_name, embedding_model, allow_dangerous_deserialization=True)
    matched_documents = loaded_index.similarity_search(user_input)

    conversation_chain = configure_chat_chain()
    return conversation_chain.invoke({"input_documents": matched_documents, "question": user_input}, return_only_outputs=True)

# Функция для запуска бота
def start_bot(api_token, pdf_file_path):
    bot = telebot.TeleBot(api_token)

    @bot.message_handler(content_types=['text'])
    def handle_message(message):
        pdf_content = read_pdf_content(pdf_file_path)
        segmented_text = partition_text(pdf_content)
        create_vector_index(segmented_text)
        model_answer = handle_user_query(message.text)
        bot.send_message(message.chat.id, model_answer['output_text'])

    bot.polling(none_stop=True, interval=0)

# Основная программа
if __name__ == "__main__":
    initialize_environment()
    API_TOKEN = "7353664838:AAHr_ja6YKLPqlVgOYHWgn5aijxfWtMSi4g"
    PDF_FILE_PATH = "text.pdf"
    start_bot(API_TOKEN, PDF_FILE_PATH)