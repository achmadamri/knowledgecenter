# Import the necessary packages
import streamlit as st
import fitz  # PyMuPDF
from langchain import YourLangChainComponent

# Define the function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
        return text

# Define the main function of the app
def main():
    st.title('AI Chat Application')

    # Text input for user message
    user_message = st.text_input('Enter your message:')

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF to extract context:", type=["pdf"])
    pdf_text = ""
    if uploaded_file is not None:
        # Extract text from the PDF and make it part of the conversation context
        pdf_text = extract_text_from_pdf(uploaded_file)

    # Integrate with langchain LLM
    if st.button('Send'):
        ai_response = YourLangChainComponent().perform_action(user_message, context=pdf_text if uploaded_file else "")
        # Displaying AI response
        st.write(ai_response)

# Run the app
if __name__ == '__main__':
    main()