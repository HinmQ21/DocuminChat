# CSV Data Query System

This app allows you to upload a CSV file, query the data, and get answers based on the contents of the CSV. The application uses language models and embeddings to provide conversational querying capabilities. It processes the CSV data, splits it into chunks, generates embeddings, and then allows users to ask questions about the data. The system is built using Streamlit, LangChain, Hugging Face, and various natural language processing (NLP) techniques.

## Features
- Upload and process CSV files.
- Split data into manageable chunks.
- Create embeddings from the data using pre-trained models.
- Use conversational AI models to answer questions based on the uploaded data.
- View data in paginated format.

## Installation

### 1. Clone the repository or download the code.
```bash
git clone <repository_url>
cd <project_directory>
```
### 2. Set up a virtual environment (optional but recommended).
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```
### 3. Install dependencies from requirements.txt.
Make sure requirements.txt is in the project directory.
```bash
pip install -r requirements.txt
```
### 4. Install model 
Install model following the instructions in file txt in models folder
## Usage
## 1. Start the application: After installing all dependencies and setting up the environment, you can run the application using Streamlit:

```bash
streamlit run app.py
```
## 2. Upload a CSV file:
Click the "Upload a CSV file" button to select and upload your CSV file.
The data will be loaded, split into chunks, and the app will show a paginated view of the CSV data.
## 3. Ask a question:
Once the data is uploaded, you can type a question in the text box.
Press the "Submit" button to get the answer based on the CSV data.
## 4. Pagination:
The data is displayed in pages to avoid overwhelming the user with too much information at once. Use the page selector to view different parts of the data.
