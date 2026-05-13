import sys, os, numpy as np
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import openai
import io
import PyPDF2
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity
from visualizer import visualize_vectors
from graph_visualizer import generate_applicant_graph, render_graph_in_streamlit

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

# Removed debug prints for security

welcome_message = """
  #### Introduction 🚀

  The system is a RAG pipeline designed to assist hiring managers in searching for the most suitable candidates out of thousands of resumes more effectively. ⚡

  The idea is to use a similarity retriever to identify the most suitable applicants with job descriptions.
  This data is then augmented into an LLM generator for downstream tasks such as analysis, summarization, and decision-making. 

  #### Getting started 🛠️

  1. To set up, please add your OpenRouter's API key. 🔑 
  2. Type in a job description query. 💬

  Hint: The knowledge base of the LLM has been loaded with a pre-existing vectorstore of [resumes](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/main-data/synthetic-resumes.csv) to be used right away. 
  In addition, you may also find example job descriptions to test [here](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/blob/main/data/supplementary-data/job_title_des.csv).

  Please make sure to check the sidebar for more useful information. 💡
"""

info_message = """
  # Information

  ### 1. What if I want to use my own resumes?

  If you want to load in your own resumes file, simply use the uploading button above. 
  Please make sure to have the following column names: `Resume` and `ID`. 

  Keep in mind that the indexing process can take **quite some time** to complete. ⌛

  ### 2. What if I want to set my own parameters?

  You can change the RAG mode and the GPT's model type using the sidebar options above. 

  About the other parameters such as the generator's *temperature* or retriever's *top-K*, I don't want to allow modifying them for the time being to avoid certain problems. 
  FYI, the temperature is currently set at `0.1` and the top-K is set at `5`.  

  ### 3. Is my uploaded data safe? 

  Your data is not being stored anyhow by the program. Everything is recorded in a Streamlit session state and will be removed once you refresh the app. 

  However, it must be mentioned that the **uploaded data will be processed directly by OpenRouter**, which I do not have control over. 
  As such, it is highly recommended to use the default synthetic resumes provided by the program. 

  ### 4. How does the chatbot work? 

  The Chatbot works a bit differently to the original structure proposed in the paper so that it is more usable in practical use cases.

  For example, the system classifies the intent of every single user prompt to know whether it is appropriate to toggle RAG retrieval on/off. 
  The system also records the chat history and chooses to use it in certain cases, allowing users to ask follow-up questions or tasks on the retrieved resumes.
"""

about_message = """
  # About

  This small program is a prototype designed out of pure interest as additional work for the author's Bachelor's thesis project. 
  The aim of the project is to propose and prove the effectiveness of RAG-based models in resume screening, thus inspiring more research into this field.

  The program is very much a work in progress. I really appreciate any contribution or feedback on [GitHub](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline).

  If you are interested, please don't hesitate to give me a star. ⭐
"""


st.set_page_config(
    page_title="Anything-RAG | Resume Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Mobile & Cross-Browser Optimization
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Mobile Responsive Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        h1 {
            font-size: 1.8rem !important;
        }
    }

    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #636EFA;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #EF553B;
        border: none;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #636EFA !important;
        color: white !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Anything-RAG")
st.caption("Advanced AI Resume Screening Assistant")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []



def parse_uploaded_files(uploaded_files):
  dfs = []
  for file in uploaded_files:
    filename = file.name
    ext = filename.split(".")[-1].lower()
    if ext == "csv":
      df = pd.read_csv(file)
      dfs.append(df)
    elif ext in ["xlsx", "xls"]:
      df = pd.read_excel(file)
      dfs.append(df)
    elif ext == "pdf":
      pdf_reader = PyPDF2.PdfReader(file)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
      df = pd.DataFrame({"ID": [filename.replace(".pdf", "")], "Resume": [text]})
      dfs.append(df)
  if dfs:
    return pd.concat(dfs, ignore_index=True)
  return None

def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  uploaded = st.session_state.uploaded_file
  
  if uploaded:
    if not isinstance(uploaded, list):
      uploaded = [uploaded]
      
    try:  
      df_load = parse_uploaded_files(uploaded)
      if df_load is None:
         raise Exception("No valid files uploaded.")
    except Exception as error:
      with modal.container():
        st.markdown("The uploaded file returns the following error message. Please check your file again.")
        st.error(error)
    else:
      if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
          st.error("Please include the following columns in your data: \"Resume\", \"ID\".")
      else:
        with st.toast('Indexing the uploaded data. This may take a while...'):
          st.session_state.df = df_load
          vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
          st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
          
          try:
            fig = visualize_vectors(
              st.session_state.rag_pipeline.vectorstore, 
              query_vector=None,
              retrieved_ids=[]
            )
            st.session_state.last_fig = fig
          except Exception as e:
            print(f"Viz error on upload: {e}")
  else:
    st.session_state.df = pd.read_csv(DATA_PATH)
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)


def check_openai_api_key(api_key: str):
  from openai import OpenAI
  client = OpenAI(
    api_key=api_key, 
    base_url=OPENROUTER_BASE_URL,
    default_headers={
      "HTTP-Referer": "https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline",
      "X-Title": "Resume Screening RAG Pipeline"
    }
  )
  try:
    _ = client.chat.completions.create(
      model="google/gemini-2.0-flash-001", 
      messages=[{"role": "user", "content": "Hello!"}],
      max_tokens=3
    )
    return True, ""
  except Exception as e:
    return False, str(e)
  
  
def check_model_name(model_name: str, api_key: str):
  from openai import OpenAI
  client = OpenAI(
    api_key=api_key, 
    base_url=OPENROUTER_BASE_URL,
    default_headers={
      "HTTP-Referer": "https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline",
      "X-Title": "Resume Screening RAG Pipeline"
    }
  )
  try:
    model_list = [model.id for model in client.models.list()]
    return True if model_name in model_list else False
  except Exception as e:
    print(f"Model name check failed: {e}")
    return False


def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

# --- API Key Security Logic ---
# Prioritize server-side environment variable (Streamlit Secrets or .env)
ENV_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Only use session state if ENV_API_KEY is missing
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Determine which key to use for logic (Backend priority)
active_api_key = ENV_API_KEY if ENV_API_KEY else st.session_state.api_key

with st.sidebar:
  st.markdown("# Control Panel")

  # Only show input field if the key is NOT in the backend environment
  if not ENV_API_KEY:
      st.info("🔑 API Key not found in server environment. Please enter it below.")
      st.text_input("OpenRouter's API Key", type="password", key="api_key")
  else:
      st.success("✅ API Key loaded from secure server environment.")
  
  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
  st.text_input("Model", "google/gemini-2.0-flash-001", key="gpt_selection")
  st.file_uploader("Upload resumes", type=["csv", "pdf", "xlsx", "xls"], accept_multiple_files=True, key="uploaded_file", on_change=upload_file)
  st.button("Clear conversation", on_click=clear_message)
  st.checkbox("Show Vector Visualization", key="show_viz", value=True)

  st.divider()
  st.markdown(info_message)

  st.divider()
  st.markdown(about_message)
  st.markdown("Made by [Hungreeee](https://github.com/Hungreeee)")


# Create tabs for different sections
tab_chat, tab_viz, tab_graph = st.tabs(["💬 Chat Assistant", "🌐 Vector Space", "📊 Graph Visualizer"])

with tab_chat:
  for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
      with st.chat_message("AI"):
        st.write(message.content)
    elif isinstance(message, HumanMessage):
      with st.chat_message("Human"):
        st.write(message.content)
    else:
      with st.chat_message("AI"):
        message[0].render(*message[1:])


if not active_api_key:
  st.info("Please add your OpenRouter API key to continue. Learn more about [OpenRouter API keys](https://openrouter.ai/keys).")
  st.stop()

is_valid, error_msg = check_openai_api_key(active_api_key)
if not is_valid:
  st.error(f"The API key is incorrect or there was a connection issue. Error: {error_msg}")
  st.stop()

if not check_model_name(st.session_state.gpt_selection, active_api_key):
  st.error("The model you specified does not exist. Learn more about [OpenRouter models](https://openrouter.ai/models).")
  st.stop()


retriever = st.session_state.rag_pipeline

llm = ChatBot(
  api_key=active_api_key,
  model=st.session_state.gpt_selection,
  base_url=OPENROUTER_BASE_URL
)

if user_query is not None and user_query != "":
  with tab_chat:
    with st.chat_message("Human"):
      st.markdown(user_query)
      st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("AI"):
      start = time.time()
      with st.spinner("Generating answers..."):
        document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
        query_type = retriever.meta_data["query_type"]
        st.session_state.resume_list = document_list
        stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
      end = time.time()

      response = st.write_stream(stream_message)
      
      retriever_message = chatbot_verbosity
      retriever_message.render(document_list, retriever.meta_data, end-start)

      st.session_state.chat_history.append(AIMessage(content=response))
      st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))

      # Generate and store visualization
      try:
        query_vector = st.session_state.embedding_model.embed_query(user_query)
        retrieved_data = retriever.meta_data.get("retrieved_docs_with_scores", {})
        retrieved_ids = retrieved_data.keys() if isinstance(retrieved_data, dict) else []
        
        st.session_state.last_fig = visualize_vectors(
          st.session_state.rag_pipeline.vectorstore, 
          query_vector=np.array(query_vector),
          retrieved_ids=retrieved_ids
        )
        
        # Generate Graph
        st.session_state.last_graph_path = generate_applicant_graph(document_list)
      except Exception as e:
        print(f"Viz error: {e}")

with tab_viz:
  st.subheader("Vector Space Analysis")
  st.write("This section visualizes how your query (green) relates to the retrieved resumes (red) among the entire dataset (blue).")
  
  if "last_fig" in st.session_state:
    st.plotly_chart(st.session_state.last_fig, use_container_width=True)
  else:
    st.info("Ask a question in the Chat Assistant tab to generate a vector visualization.")

with tab_graph:
  st.subheader("Knowledge Graph Analysis")
  st.write("This graph shows the connections between retrieved applicants and their shared skills.")
  
  if st.button("Generate Graph for Uploaded Materials"):
    with st.spinner("Generating Knowledge Graph..."):
      docs_with_id = ["Applicant ID " + str(row['ID']) + "\\n" + str(row['Resume']) for _, row in st.session_state.df.iterrows()]
      st.session_state.last_graph_path = generate_applicant_graph(docs_with_id)
      
  if "last_graph_path" in st.session_state:
    render_graph_in_streamlit(st.session_state.last_graph_path)
  else:
    st.info("Click the button above to generate a graph for the current resumes, or ask a question in the Chat Assistant tab.")