# Audio-based Question-Answering System

This project is a hybrid system that captures audio input, processes it to generate a text transcription, analyzes the transcription, and produces an appropriate response using an LLM (Llama model) integrated with Langchain. The final response is then converted back to speech. The project involves a Flask web app for easy deployment and an accompanying notebook for testing the pipeline.

### Features
- **Audio to Text Conversion**: Uses OpenAI's Whisper model for high-quality speech-to-text transcription.
- **Text-based Question Answering**: Leverages the Langchain integration with the Hugging Face Llama model to provide intelligent responses.
- **Text to Speech Response**: Converts the response back into audio using Google Text-to-Speech (gTTS).

### Technologies Used
- **Flask**: For serving the web application.
- **Langchain** and **Hugging Face Hub**: To enable natural language processing and responses.
- **gTTS**: For converting text responses to speech audio files.
- **Librosa** and **pydub**: For audio processing and silence detection.
- **OpenAI Whisper**: Used for converting audio input to text.

### Workflow

The system workflow is designed to convert audio input into a meaningful response and play it back to the user:

![Untitled Diagram](https://github.com/user-attachments/assets/7ff68eb4-778d-4534-a54f-73658ea1ec99)


1. **Audio Input** (MP3/WAV): User provides an audio file containing the question.
2. **Whisper Model for Transcription**: Converts audio to text, extracting the question.
3. **Llama Model with Langchain for Response**: Uses the extracted question to generate an intelligent response.
4. **Google Text-to-Speech (gTTS)**: Converts the response back to audio.
5. **Audio Playback**: Provides an audio response to the user.

### Project Structure
- `main.py`: Flask application file that sets up the server, processes incoming audio files, and serves responses.
- `Chatbot_using_Whisper.ipynb`: Jupyter Notebook for testing the pipeline with individual audio files and direct model interactions.
- `templates`: Contain HTML for the app's user interface.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. **Install Required Packages**:
   ```bash
   pip install flask langchain langchain_community transformers gtts librosa pydub
   ```

3. **Download Necessary Models**:
   - Whisper model (`openai/whisper-small.en`)
   - Llama model (`meta-llama/Llama-3.2-3B-Instruct`)
   - Ensure you have a Hugging Face API token.

4. **Run Flask App**:
   ```bash
   python app.py
   ```

5. **Notebook Execution**: 
   Use the notebook (`Chatbot_using_Whisper.ipynb`) to test the pipeline end-to-end in an interactive environment.

### Usage

1. **Web Application**:
   - Navigate to `http://127.0.0.1:5000/` in your browser.
   - Upload an audio file to ask a question, and the app will provide an audio response.

2. **Notebook Workflow**:
   - Import necessary libraries and initialize models.
   - Input your audio file path, convert it to text, generate a response using the model, and convert the text back to an audio file.

### Example Walkthrough

1. **Audio Input**: Provide an audio file containing your question.
2. **Text Conversion**: `Whisper` transcribes the audio to text.
3. **Response Generation**: `Llama` provides a response based on the transcription.
4. **Text to Speech**: `gTTS` converts the response to audio.

### Code Explanation

#### Flask Web App (`app.py`)
- **Route Handlers**:
  - `/process_audio`: Accepts audio files, processes them to generate responses.
  - `add_header`: Caches and response handling.
- **Audio Processing**:
  - Uses `pydub` to handle audio files, Whisper for transcription, and Langchain LLM for responses.
  
#### Notebook (`Chatbot_using_Whisper.ipynb`)
- **Speech Recognition**: Uses Whisper model to transcribe audio files.
- **Language Model**: Langchain integration with the Llama model for question-answering.
- **Text to Speech**: `gTTS` to generate audio files for responses.

### Configuration
- **Hugging Face API**: Set up your `huggingface_api_token` in the code for secure model access.
- **Model Configurations**: Adjust parameters like `temperature` and `max_new_tokens` for custom response generation.

### Acknowledgements
- **Hugging Face Hub** and **Langchain Community** for providing robust NLP pipelines.
- **Google Text-to-Speech (gTTS)** for simple text-to-speech conversion.
- **OpenAI** for Whisper model providing high-quality speech recognition.

### License
This project is licensed under the MIT License.
```
