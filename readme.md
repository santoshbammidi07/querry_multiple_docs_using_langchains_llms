# Multiple PDF Chat Question and Answering App
---------------------------------------------

## Retrieval Augmentation Generation with LLMs ( Generative AI - Document Retrieval and Question Answering).

Author: Santosh Kumar Bammidi

[Github Link](https://github.com/santoshbammidi07/querry_multiple_docs_using_langchains_llms.git)
-
Date: 18/07/2023

## Introduction
----------------
This App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes large language model to generate accurate answers to your queries.

Please note that the app will only respond to questions related to the loaded PDFs..

## How Does it work 
--------------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. **Load PDF**: The app reads multiple PDF documents and extracts their text content.

2. **Text Chunking**: The extracted text is divided into smaller chunks that can be processed effectively.
3. **Embeddingd using Language Model**: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. **Similarity Matching**: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. **Response Generation**: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs..

## Dependencies and Installation
----------------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. **Install the required dependencies by running the following command**:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory..

## Usage
-----------
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file..

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by clicking on browse.

5. Ask questions in natural language regarding the content present in the uploaded PDFs using the chat interface.






from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import json

uploaded_file = client.files.upload(
    file={
        "file_name": pdf_file.stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
)

signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

pdf_response = client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=True)

response_dict = json.loads(pdf_response.json())
json_string = json.dumps(response_dict, indent=4)
print(json_string)
