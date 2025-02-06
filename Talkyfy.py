import os
import wave
import pyaudio
import whisper
import streamlit as st
from gtts import gTTS
import pygame
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load Whisper model
model = whisper.load_model("base")

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 7
WAVE_OUTPUT_FILENAME = "input.wav"

def record_audio():
    """Records a short audio clip and saves it to a file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    st.info("🎤 Recording...")
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    st.success("✅ Recording complete!")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio():
    """Transcribes recorded audio using Whisper."""
    st.info("📝 Transcribing...")
    result = model.transcribe(WAVE_OUTPUT_FILENAME)
    return result['text']

def generate_ai_response(user_input, llm):
    """Uses Groq LLM to generate a response based on user input."""
    prompt = PromptTemplate(
        input_variables=["text"],
        template="You are an AI assistant. Respond to the following user query concisely in 5 to 10 lines: {text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(user_input)

def text_to_speech(text):
    """Converts text to speech and plays it."""
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.quit()
    os.remove("response.mp3")

# Streamlit UI
st.set_page_config(page_title="Talkify", layout="centered")
st.markdown("<h1 style='text-align: center; color: darkblue;'>🔊 Talkify</h1>", unsafe_allow_html=True)
st.write("🎙️ Speak into the microphone and get an AI-generated response!")

# User enters Groq API Key
groq_api_key = st.text_input("🔑 Enter your GROQ API Key:", type="password")

# User selects the AI model
model_options = [
    "llama-3.2-3b-preview",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "llama-3.3-70b-specdec",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile"
]
selected_model = st.selectbox("🤖 Choose an AI Model:", model_options, index=1)  # Default: deepseek-r1-distill-llama-70b

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)

    if st.button("🎤 Start Recording"):
        record_audio()
        user_text = transcribe_audio()
        
        if user_text:
             # Display the recorded audio file and provide download link
            st.audio(WAVE_OUTPUT_FILENAME, format='audio/wav', start_time=0)
            st.download_button(
                label="Download Recorded Audio",
                data=open(WAVE_OUTPUT_FILENAME, "rb").read(),
                file_name=WAVE_OUTPUT_FILENAME,
                mime="audio/wav"
            )
            st.success(f"**You said:** {user_text}")

            # AI-generated response using Groq's LLM
            with st.spinner(f"🤖 Generating response using {selected_model}..."):
                response = generate_ai_response(user_text, llm)
            
            st.success(f"🤖 **AI Response:** {response}")

            # Speak the AI response
            text_to_speech(response)

           
