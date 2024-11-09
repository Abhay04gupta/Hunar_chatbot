# Audio-Based Q&A Chatbot with Langchain and Hugging Face

This project is an audio-based Question and Answer (Q&A) chatbot. Users can submit a spoken question in the form of an audio file, and the system will convert the audio into text, generate a response using a language model, and then convert the response back into audio for playback. It leverages the following technologies:
- **Langchain** for managing the language model prompt and response generation.
- **Hugging Face Transformers** for automatic speech recognition and text-based language processing.
- **Google Text-to-Speech (gTTS)** for converting the response text back into audio.

This chatbot is designed for seamless audio input and output, providing an interactive, voice-based experience.

## Project Workflow

The data workflow is organized into six steps, converting an audio query to a generated audio response as follows:

### Workflow Overview

1. **Audio Input**: 
   - User uploads an audio file containing a spoken question.

2. **Audio to Text Conversion**: 
   - The audio file is transcribed into text using OpenAI's Whisper model (`openai/whisper-small.en`) from Hugging Face. 

3. **Question Extraction**:
   - The transcribed text (question) is extracted for further processing.

4. **Model Response Generation**:
   - The extracted question text is passed to a language model (Llama 3.2 3B model), where a response is generated using Langchain’s `LLMChain`.

5. **Text to Speech Conversion**:
   - The model's response text is converted into speech using Google Text-to-Speech (gTTS).

6. **Response Playback**:
   - The generated response is saved as an audio file and played back to the user.

### Detailed Workflow Diagram

