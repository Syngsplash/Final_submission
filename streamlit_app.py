import streamlit as st
import pandas as pd
from groq import Groq
from datetime import datetime
import io
import PyPDF2
from PIL import Image
import pytesseract
import os
from pathlib import Path

tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

csv_path = Path(__file__).parent / 'data/job skill context(Core competencies).csv'

@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)

def main():
    st.title("Resume Assistant Chatbot")

    # Load the job skills CSV file internally
    df = load_data()

    if df is None:
        st.error("Failed to load the job skills data. Please check the file path.")
        st.stop()

    # File uploader for resume (now supports PDF, TXT, and images)
    uploaded_resume = st.file_uploader("Upload your resume", type=["pdf", "txt", "png", "jpg", "jpeg"])
    
    # Occupation selector
    unique_occupations = df['ANZSCO Title'].unique()
    selected_occupation = st.selectbox("Select your occupation", unique_occupations)

    # Get skills for selected occupation
    occupation_skills = df[df['ANZSCO Title'] == selected_occupation]
    skills_text = "\n".join([f"{row['Core Competency']}: {row['Anchor Value']}" for _, row in occupation_skills.iterrows()])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your resume or skills:"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if uploaded_resume is not None:
            # Extract text based on file type
            file_type = uploaded_resume.type
            if file_type == "application/pdf":
                resume_content = extract_text_from_pdf(uploaded_resume)
            elif file_type.startswith("image/"):
                resume_content = extract_text_from_image(uploaded_resume)
            else:  # Assume it's a text file
                resume_content = uploaded_resume.getvalue().decode("utf-8")
            
            #api key call
            groq_api_key = "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"
            client=Groq(
                api_key=st.secrets["GROQ_API_KEY"],
            )
            completion = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": f"Help a user to understand how to gain soft skills for their resume. Here is the resume:{resume_content}\nHere are the skills required for the occupation '{selected_occupation}':\n{skills_text}\n\nUser question: {prompt}"
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(completion.choices[0].message.content)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": completion.choices[0].message.content})

            # Save the response to a file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file_name = f'resume_{timestamp}.md'
            with open(output_file_name, 'w') as output_file:
                output_file.write(completion.choices[0].message.content)
            
            st.success(f"Response saved to {output_file_name}")
        else:
            st.error("Please upload a resume.")

    # Display uploaded resume (if it's an image)
    if uploaded_resume is not None and uploaded_resume.type.startswith("image/"):
        st.image(uploaded_resume, caption="Uploaded Resume", use_column_width=True)

if __name__ == "__main__":
    main()