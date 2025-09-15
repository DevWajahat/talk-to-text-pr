import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import wave
import io
import numpy as np
import resampy
from tqdm import tqdm
from fpdf import FPDF, HTMLMixin
from mistletoe import markdown
import json
import time
import requests
import re
import firebase_admin
from firebase_admin import credentials, auth, db
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# ---------------- Firebase setup ----------------
# NOTE: Replace with your actual Firebase config in a secrets file.
# The following is a placeholder for demonstration purposes.
firebase_config = os.path.join(".streamlit","firebase-config.json")

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://talktotext-61c7b-default-rtdb.firebaseio.com/"
        })
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.info("Please make sure your Firebase configuration is correct.")
        
def save_transcription_history(uid, filename, transcription, summary):
    try:
        ref = db.reference(f'users/{uid}/transcriptions')
        new_ref = ref.push()
        new_ref.set({
            "filename": filename,
            "raw_transcription": transcription,
            "translated_summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def get_user_transcriptions(uid):
    try:
        ref = db.reference(f'users/{uid}/transcriptions')
        return ref.get()
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return None

def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.user_id = user.uid
        st.session_state.page = "app"
        st.success("Login Successful!")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

def signup_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        st.session_state.user_id = user.uid
        st.session_state.page = "app"
        st.success("Signup Successful! You are now logged in.")
        st.rerun()
    except Exception as e:
        st.error(f"Signup failed: {e}")

def logout_user():
    st.session_state.user_id = None
    st.session_state.page = "login"
    st.rerun()

# ---------------- Gemini API ----------------
def get_gemini_response(prompt_text, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'x-goog-api-key': api_key
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt_text
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            return response_json['candidates'][0]['content']['parts'][0]['text']
        else:
            return "No content generated."
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# --- Setup for PDF Generation ---
class PDF(FPDF, HTMLMixin):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'TalkToText Pro: Meeting Notes', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def parse_markdown_to_pdf(pdf_obj, markdown_text):
    pdf_obj.set_font('DejaVuSans', '', 10)
    lines = markdown_text.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('## '):
            pdf_obj.set_font('DejaVuSans', 'B', 14)
            pdf_obj.multi_cell(0, 10, stripped_line[3:])
            pdf_obj.ln(2)
            pdf_obj.set_font('DejaVuSans', '', 10)
        elif stripped_line.startswith(('* ', '- ')):
            pdf_obj.set_font('DejaVuSans', '', 10)
            pdf_obj.cell(10, 5, '•')
            pdf_obj.multi_cell(0, 5, stripped_line[2:])
            pdf_obj.ln(1)
        else:
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    pdf_obj.set_font('DejaVuSans', 'B', 10)
                    pdf_obj.write(5, part[2:-2])
                    pdf_obj.set_font('DejaVuSans', '', 10)
                elif part.startswith('*') and part.endswith('*'):
                    pdf_obj.set_font('DejaVuSans', 'I', 10)
                    pdf_obj.write(5, part[1:-1])
                    pdf_obj.set_font('DejaVuSans', '', 10)
                else:
                    pdf_obj.write(5, part)
            pdf_obj.ln(5)

def create_pdf(filename, transcription, summary):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", size=12)

    title_html = markdown(f'# Transcription and Summary for "{filename or "Untitled"}"')
    pdf.write_html(title_html)

    transcription_html = markdown('## Transcription:\n\n' + transcription)
    pdf.write_html(transcription_html)
    pdf.ln(10)

    summary_html = markdown('## Structured Notes:\n\n' + summary)
    pdf.write_html(summary_html)

    safe_filename = filename or "untitled"
    pdf_dir = "pdf_outputs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{os.path.splitext(safe_filename)[0]}.pdf")

    pdf.output(pdf_path)
    return pdf_path

# --- Initialize Streamlit Session State ---
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'blog_post' not in st.session_state:
    st.session_state.blog_post = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'first_chat_message' not in st.session_state:
    st.session_state.first_chat_message = True

# --- Initialize Hugging Face Models ---
@st.cache_resource
def load_whisper_model():
    st.write("Loading Whisper model... This may take a moment. ☕")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    st.success("Successfully loaded pre-trained Whisper model!")
    return processor, model

# --- Streamlit UI ----------------
st.set_page_config(page_title="TalkToText Pro", layout="wide")

if st.session_state.page == "login":
    st.title("TalkToText Pro - Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        login_user(email, password)
    if st.button("Create Account"):
        st.session_state.page = "signup"
        st.rerun()

elif st.session_state.page == "signup":
    st.title("TalkToText Pro - Signup")
    st.markdown(
        """
        <p style="font-size: 16px; color: #ccc; text-align: center;">
            `Before you can start using TalkToText Pro, you need to create an account.
            Simply enter your desired email and password below to get started!`
        </p>
        """,
        unsafe_allow_html=True
    )
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        signup_user(email, password)
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

elif st.session_state.page == "app" and st.session_state.user_id:
    st.title("TalkToText Pro 🎙️✨")
    st.sidebar.button("Logout", on_click=logout_user)

    st.markdown(
        """
        <div style="text-align: center; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #4CAF50;">Welcome to TalkToText Pro! 🎙️</h2>
            <p style="font-size: 16px; color: #555;">
                TalkToText Pro is an advanced AI-powered application designed to streamline your meeting workflow. 
                Simply upload an audio file and our system will generate a detailed transcription, 
                a structured summary with key decisions and action items, and even a professionally written blog post. 
                You can export all of this information as a downloadable PDF report, and use our integrated chatbot 
                to ask questions and get instant answers about the meeting content.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("How to Use this Web App"):
        st.markdown("""
        To use this application, follow these simple steps:
        * **1. Upload Audio:** In the "Upload Audio File" section, select a `.wav` file from your local machine.
        * **2. Process Audio:** Click the "Process Audio" button. Our AI models will transcribe the audio and generate a structured summary.
        * **3. Review Notes:** The full transcript and structured notes will appear on the right side of the screen.
        * **4. Generate & Export:** Use the "Generate PDF" button to download a complete report of the meeting. You can also generate a blog post using the "Generate Blog Post" section below.
        * **5. Ask Questions:** Use the chatbot window at the bottom-right of the screen to ask questions about the meeting content.
        """)

    st.sidebar.header("Transcription History")
    history = get_user_transcriptions(st.session_state.user_id)
    if history:
        for k, record in history.items():
            if st.sidebar.button(f"{record['filename']} ({record['timestamp']})"):
                st.session_state.transcription = record['raw_transcription']
                st.session_state.summary = record['translated_summary']
    else:
        st.sidebar.info("No transcription history found.")
    
    tab1, tab2, tab3, tab4  = st.tabs(["Meeting Transcription & Summary", "Blog Post", "Chat with Gemini", "Technologies Used"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.header("Upload Audio File 📁")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav'],
                help="Only uncompressed WAV files are supported. They will be automatically resampled to 16,000 Hz."
            )
        
        with col2:
            st.header("Meeting Transcription & Summary 📝")
            
            if uploaded_file:
                st.audio(uploaded_file, format='audio/wav')
                
                if uploaded_file.name != st.session_state.uploaded_file_name:
                    st.session_state.transcription = ""
                    st.session_state.summary = ""
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                if st.button("Process Audio"):
                    st.session_state.transcription = ""
                    st.session_state.summary = ""
                    with st.spinner("Processing audio..."):
                        try:
                            processor, whisper_model = load_whisper_model()
                            
                            audio_bytes_io = io.BytesIO(uploaded_file.getvalue())
                            
                            with wave.open(audio_bytes_io, 'rb') as wave_file:
                                n_channels = wave_file.getnchannels()
                                sample_rate = wave_file.getframerate()
                                n_frames = wave_file.getnframes()
                                audio_frames = wave_file.readframes(n_frames)
                            
                            audio_data = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32) / 32768.0

                            if n_channels > 1:
                                st.info("Stereo audio detected. Converting to mono for transcription...")
                                stereo_data = audio_data.reshape(-1, n_channels)
                                mono_audio = np.mean(stereo_data, axis=1)
                            else:
                                mono_audio = audio_data

                            if sample_rate != 16000:
                                st.info(f"Resampling audio from {sample_rate} Hz to 16,000 Hz...")
                                resampled_audio = resampy.resample(mono_audio, sr_orig=sample_rate, sr_new=16000)
                            else:
                                resampled_audio = mono_audio

                            st.subheader("Raw Transcription")
                            
                            total_duration_s = len(resampled_audio) / 16000
                            chunk_length_s = 30
                            overlap_s = 5
                            total_chunks = int(np.ceil(total_duration_s / (chunk_length_s - overlap_s)))
                            full_transcription = []
                            
                            progress_bar = st.progress(0, text="Transcribing audio...")
                            
                            for i in tqdm(range(total_chunks)):
                                start_s = i * (chunk_length_s - overlap_s)
                                end_s = start_s + chunk_length_s
                                start_idx = int(start_s * 16000)
                                end_idx = int(end_s * 16000)
                                audio_chunk = resampled_audio[start_idx:end_idx]
                                if len(audio_chunk) == 0:
                                    continue

                                input_features = processor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_features
                                predicted_ids = whisper_model.generate(input_features)
                                transcription_chunk = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                                
                                if "I'm not a human being" in transcription_chunk.strip():
                                    st.warning("Filtered out a hallucinated transcription chunk.")
                                    continue
                                
                                full_transcription.append(transcription_chunk)
                                progress_bar.progress((i + 1) / total_chunks, text=f"Processing chunk {i+1} of {total_chunks}")
                            
                            final_transcript = " ".join(full_transcription)
                            st.session_state.transcription = final_transcript
                            
                            st.subheader("Structured Meeting Notes")
                            with st.spinner("Generating structured notes with Gemini..."):
                                gemini_prompt = f"""
                                You are a professional meeting assistant. Your task is to analyze the following audio transcription of a meeting and generate comprehensive, structured meeting notes and also translate transcription in english if there in any other language.

                                The output should be a single markdown document that includes:

                                ## Table of Contents
                                Extract the most important keywords and topics and present them as a clickable table of contents for easy navigation. Use bullet points for this list.

                                ## Executive Summary
                                A 3-4 sentence summary of the key discussion points and outcomes.

                                ## Key Discussion Points
                                * A bullet-point list summarizing the main topics covered.
                                
                                ## Decisions 
                                give desicions what you see in the meeting transcription
                                
                                ## Action Items
                                give action items what you see in the meeting transcription
                                
                                ##Sentiment 
                                (positive/negative tone)
                                
                                ##Full Transcription with speaker recognition (translate in english if any other language)
                                eg:-
                                [Speaker:1]dasfadsf
                                [Speaker:2]dasfdas
                                
                                Here is the meeting transcription:
                                
                                {final_transcript}
                                """
                                
                                gemini_api_key = st.secrets["GEMINI_API_KEY"]
                                summary = get_gemini_response(gemini_prompt, gemini_api_key)
                                
                                if summary:
                                    st.session_state.summary = summary
                                    save_transcription_history(st.session_state.user_id, uploaded_file.name, final_transcript, summary)
                                else:
                                    st.session_state.summary = "Summary generation failed. Please check the API key and try again."
                        
                        except Exception as e:
                            st.error(f"An error occurred during processing: {e}")
                            st.info("Please ensure the WAV file is uncompressed and valid.")
                            st.session_state.transcription = ""
                            st.session_state.summary = ""
        
        if st.session_state.transcription:
            st.text_area("Full Transcript", st.session_state.transcription, height=200)
            
            st.subheader("Structured Meeting Notes")
            st.markdown(st.session_state.summary)

            st.markdown("---")
            st.subheader("Export to PDF 📄")
            if st.button("Generate PDF"):
                if st.session_state.transcription and st.session_state.summary:
                    pdf_path = create_pdf(st.session_state.uploaded_file_name, st.session_state.transcription, st.session_state.summary)
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_file,
                            file_name=os.path.basename(pdf_path),
                            mime="application/octet-stream"
                        )
                    st.success("PDF report generated and ready for download!")
                else:
                    st.error("Please process the audio first to generate the transcription and summary.")

    with tab2:
        st.subheader("Generate Blog Post ✍️")
        
        word_count = st.number_input(
            "Maximum word count (max 2000)", 
            min_value=100, 
            max_value=2000, 
            value=500,
            step=50
        )
        seo_keywords = st.text_input(
            "Enter SEO keywords (comma-separated)",
            placeholder="e.g., meeting notes, AI transcription, productivity"
        )
        
        if st.button("Generate Blog Post"):
            if st.session_state.summary:
                with st.spinner("Generating blog post with Gemini..."):
                    blog_prompt = f"""
                    You are a professional blog post writer. Your task is to create a compelling and informative blog post based on the following meeting notes and summary.

                    Requirements:
                    - The blog post should be approximately {word_count} words long.
                    - It must incorporate the following SEO keywords: {seo_keywords}.
                    - The tone should be professional and engaging.
                    - Structure the blog post with a title, a clear introduction, body paragraphs with subheadings, and a conclusion.
                    - Use the provided meeting summary and full transcription as your primary source of information.

                    Here is the meeting summary and notes:
                    
                    {st.session_state.summary}

                    Please write the full blog post.
                    """
                    
                    gemini_api_key = st.secrets["GEMINI_API_KEY"]
                    blog_post_content = get_gemini_response(blog_prompt, gemini_api_key)
                    
                    if blog_post_content:
                        st.session_state.blog_post = blog_post_content
                        st.success("Blog post generated successfully!")
                    else:
                        st.error("Blog post generation failed. Please try again.")
            else:
                st.error("Please process the audio first to generate the meeting notes.")
                
        if st.session_state.blog_post:
            st.subheader("Generated Blog Post")
            st.markdown(st.session_state.blog_post)
            
    with tab3:
        st.header("Chat with Gemini about your Meeting")
        st.info("Ask me anything about the transcribed meeting! I can answer questions based on the full transcript and summary.")

        # --- Chatbot UI ---
        st.markdown(
            """
            <style>
            .chat-container {
                height: 450px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
            .chat-message {
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 8px;
            }
            /* Updated styling for user messages */
            .chat-message.user {
                background-color: #555;
                color: #fff; /* White text color */
                text-align: right;
            }
            /* Updated styling for assistant messages */
            .chat-message.assistant {
                background-color: #333;
                color: #fff; /* White text color */
                text-align: left;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Initialize chat_input_key if it doesn't exist
        if 'chat_input_key' not in st.session_state:
            st.session_state.chat_input_key = 0

        with st.container():
            with st.container(height=450):
                if st.session_state.first_chat_message:
                    st.session_state.chat_history.append(("assistant", "Hello! I am your meeting assistant. Ask me anything about the meeting minutes."))
                    st.session_state.first_chat_message = False

                for role, message in st.session_state.chat_history:
                    if role == "user":
                        st.markdown(f'<div class="chat-message user">{message}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-message assistant">{message}</div>', unsafe_allow_html=True)

            user_input = st.text_input("Ask a question about the meeting...", key=f"chat_input_{st.session_state.chat_input_key}", label_visibility="collapsed")
            
            if user_input:
                st.session_state.chat_history.append(("user", user_input))
                with st.spinner("Thinking..."):
                    chat_prompt = f"""
                    You are a helpful assistant specialized in answering questions about meeting minutes.
                    Here is the meeting summary and the full transcription. Use this information to answer the user's question accurately.

                    Meeting Summary:
                    {st.session_state.summary}

                    Full Transcription:
                    {st.session_state.transcription}

                    User's Question:
                    {user_input}

                    Answer the question based *only* on the provided meeting content.
                    """
                    
                    gemini_api_key = st.secrets["GEMINI_API_KEY"]
                    bot_response = get_gemini_response(chat_prompt, gemini_api_key)

                    st.session_state.chat_history.append(("assistant", bot_response))
                    
                    # Increment the key to reset the input field
                    st.session_state.chat_input_key += 1
                    st.rerun()

                                        
    with tab4:
        st.header("Technologies Used")
        st.markdown("""
        This application is built using a powerful combination of open-source libraries and APIs to provide a seamless and intelligent user experience. Here are the key technologies behind TalkToText Pro:

        ### **1. Core Framework: Streamlit**
        * **Purpose:** Streamlit is the foundation of this web application. It's a Python library that allows us to create beautiful, interactive, and data-driven web apps with very little code.
        * **Role:** It handles the entire user interface, from file uploads and button clicks to displaying the transcription, summary, and blog post.

        ### **2. Audio Transcription: Hugging Face & Whisper**
        * **Purpose:** To convert the audio file into text.
        * **Role:** We use the `transformers` library from Hugging Face, specifically leveraging the **Whisper** model developed by OpenAI. The Whisper model is a state-of-the-art speech recognition system that is highly effective for transcribing audio.

        ### **3. AI Summarization & Content Generation: Google Gemini**
        * **Purpose:** To transform the raw transcription into a structured summary and a professional blog post.
        * **Role:** The **Google Gemini API** is used to analyze the transcribed text and generate intelligent, well-structured meeting notes (including key decisions and action items) and a compelling blog post based on user-specified keywords and length.

        ### **4. User Authentication & Data Storage: Firebase**
        * **Purpose:** To securely manage user accounts and save transcription history.
        * **Role:** **Firebase Authentication** handles the login and sign-up processes, while the **Firebase Realtime Database** stores each user's transcription history so they can access it later.

        ### **5. PDF Generation: FPDF**
        * **Purpose:** To create downloadable PDF reports of the meeting notes.
        * **Role:** The `fpdf` library is used to programmatically generate a PDF document containing the file name, full transcription, and the structured meeting notes. The `mistletoe` library is used to parse the Markdown output from Gemini into HTML for better formatting in the PDF.

        ### **6. Audio Processing: NumPy & resampy**
        * **Purpose:** To prepare the audio file for the transcription model.
        * **Role:** **NumPy** is used for numerical operations on the audio data, and **resampy** is used to resample the audio to the required 16,000 Hz sample rate for the Whisper model.
        """)

 

    # # --- Chatbot Window ---
    # st.markdown(
    #     """
    #     <style>
    #     .chat-container {
    #         position: fixed;
    #         bottom: 20px;
    #         right: 20px;
    #         width: 350px;
    #         height: 450px;
    #         background-color: white;
    #         border: 1px solid #ccc;
    #         border-radius: 10px;
    #         box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    #         display: flex;
    #         flex-direction: column;
    #         z-index: 1000;
    #     }
    #     .chat-header {
    #         background-color: #f0f2f6;
    #         padding: 10px;
    #         border-top-left-radius: 10px;
    #         border-top-right-radius: 10px;
    #         font-weight: bold;
    #         text-align: center;
    #     }
    #     .chat-messages {
    #         flex-grow: 1;
    #         overflow-y: auto;
    #         padding: 10px;
    #     }
    #     .chat-input {
    #         padding: 10px;
    #         border-top: 1px solid #ccc;
    #     }
    #     .chat-message.user {
    #         background-color: #dcf8c6;
    #         border-radius: 10px;
    #         padding: 8px 12px;
    #         margin-bottom: 8px;
    #         align-self: flex-end;
    #     }
    #     .chat-message.assistant {
    #         background-color: #f0f2f6;
    #         border-radius: 10px;
    #         padding: 8px 12px;
    #         margin-bottom: 8px;
    #         align-self: flex-start;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    # with st.container():
    #     st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    #     st.markdown('<div class="chat-header">Meeting Chatbot</div>', unsafe_allow_html=True)
    #     st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
    #     for role, message in st.session_state.chat_history:
    #         if role == "user":
    #             st.markdown(f'<div class="chat-message user">{message}</div>', unsafe_allow_html=True)
    #         else:
    #             st.markdown(f'<div class="chat-message assistant">{message}</div>', unsafe_allow_html=True)

    #     st.markdown('</div>', unsafe_allow_html=True)

    #     with st.container():
    #         st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    #         user_input = st.text_input("Ask a question about the meeting...", key="chat_input", label_visibility="collapsed")
    #         st.markdown('</div>', unsafe_allow_html=True)
        
    #     if st.session_state.first_chat_message:
    #         st.session_state.chat_history.append(("assistant", "Hello! I am your meeting assistant. Ask me anything about the meeting minutes."))
    #         st.session_state.first_chat_message = False

    #     if user_input:
    #         st.session_state.chat_history.append(("user", user_input))
    #         with st.spinner("Thinking..."):
    #             chat_prompt = f"""
    #             You are a helpful assistant specialized in answering questions about meeting minutes.
    #             Here is the meeting summary and the full transcription. Use this information to answer the user's question accurately.

    #             Meeting Summary:
    #             {st.session_state.summary}

    #             Full Transcription:
    #             {st.session_state.transcription}

    #             User's Question:
    #             {user_input}

    #             Answer the question based *only* on the provided meeting content.
    #             """
                
    #             gemini_api_key = st.secrets["GEMINI_API_KEY"]
    #             bot_response = get_gemini_response(chat_prompt, gemini_api_key)

    #             st.session_state.chat_history.append(("assistant", bot_response))
    #             st.rerun()

    #     st.markdown('</div>', unsafe_allow_html=True)


st.markdown("Powered by Streamlit and open-source models from Hugging Face and Gemini.")