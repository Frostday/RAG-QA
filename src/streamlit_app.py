"""Streamlit frontend for the Question-Answering Bot API."""
import json
import sys
from pathlib import Path

import requests
import streamlit as st

# Add src directory to path for imports
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import configuration
from config import (
    MAX_FILE_SIZE_MB,
    MAX_QUESTIONS,
    REQUEST_TIMEOUT_SECONDS,
    API_URL,
    SUPPORTED_DOCUMENT_FORMATS,
)

# Page configuration
st.set_page_config(
    page_title="Question-Answering Bot",
    page_icon="🤖",
    layout="wide"
)

# Default API URL (can be overridden by user)
api_url = API_URL

# Title and description
st.title("Question-Answering Bot")
st.markdown("Upload a document and questions to get AI-powered answers based on your document content.")

# Display limits and constraints
st.info(f"""
**📋 Limits & Requirements:**
- **Maximum file size:** {MAX_FILE_SIZE_MB} MB per document
- **Maximum questions:** {MAX_QUESTIONS} questions per request
- **Supported document formats:** {', '.join([fmt.upper().replace('.', '') for fmt in SUPPORTED_DOCUMENT_FORMATS])}
- **Questions file format:** JSON only
- **Processing timeout:** {REQUEST_TIMEOUT_SECONDS // 60} minutes
""")

# Add expandable section with error handling info
with st.expander("ℹ️ Error Handling & Tips"):
    st.markdown("""
    **Common Issues & Solutions:**
    
    - **File too large?** Try splitting your document into smaller sections
    - **Processing timeout?** Reduce the number of questions or use a smaller document
    - **Invalid format?** Ensure your questions file is valid JSON: `["question1", "question2"]`
    - **Empty results?** Check that your document contains relevant text content
    
    **Error Messages:**
    - Clear validation errors for invalid files or formats
    - Helpful suggestions when processing fails
    - Automatic cleanup of temporary files
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Document")
    document_file = st.file_uploader(
        "Choose document file",
        type=[fmt.replace('.', '') for fmt in SUPPORTED_DOCUMENT_FORMATS],
        help=f"Upload a {' or '.join([fmt.upper() for fmt in SUPPORTED_DOCUMENT_FORMATS])} document (max {MAX_FILE_SIZE_MB} MB)"
    )
    
    if document_file:
        file_size_mb = document_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File too large: {file_size_mb:.2f} MB (max {MAX_FILE_SIZE_MB} MB)")
        else:
            st.success(f"✅ {document_file.name} uploaded")
            if file_size_mb < 1:
                st.caption(f"File size: {document_file.size / 1024:.2f} KB")
            else:
                st.caption(f"File size: {file_size_mb:.2f} MB")

with col2:
    st.subheader("❓ Upload Questions")
    questions_file = st.file_uploader(
        "Choose questions file",
        type=["json"],
        help=f"Upload a JSON file containing questions (max {MAX_QUESTIONS} questions)"
    )
    
    if questions_file:
        # Preview questions file
        try:
            questions_content = questions_file.read()
            questions_data = json.loads(questions_content)
            
            if isinstance(questions_data, list):
                questions_list = questions_data
            elif isinstance(questions_data, dict) and "questions" in questions_data:
                questions_list = questions_data["questions"]
            else:
                questions_list = []
            
            if len(questions_list) > MAX_QUESTIONS:
                st.error(f"❌ Too many questions: {len(questions_list)} (max {MAX_QUESTIONS})")
            elif len(questions_list) == 0:
                st.warning("⚠️ No questions found in file")
            else:
                st.success(f"✅ {questions_file.name} uploaded")
                st.caption(f"Found {len(questions_list)} question(s)")
            
            # Show preview
            with st.expander("Preview Questions"):
                for i, q in enumerate(questions_list[:10], 1):  # Show first 10
                    st.markdown(f"{i}. {q}")
                if len(questions_list) > 10:
                    st.caption(f"... and {len(questions_list) - 10} more questions")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file")
        except Exception as e:
            st.error(f"❌ Error reading questions file: {str(e)}")

# Process button
st.markdown("---")
process_button = st.button(
    "🚀 Process Documents",
    type="primary",
    use_container_width=True,
    disabled=(document_file is None or questions_file is None)
)

# Results section
if process_button:
    if not document_file or not questions_file:
        st.error("❌ Please upload both document and questions files")
    else:
        with st.spinner("🔄 Processing documents and generating answers..."):
            try:
                # Prepare files for upload
                files = {
                    "document": (document_file.name, document_file.getvalue(), document_file.type),
                    "questions_file": (questions_file.name, questions_file.getvalue(), "application/json")
                }
                
                # Make API request
                endpoint = f"{api_url}/process-documents"
                response = requests.post(endpoint, files=files, timeout=REQUEST_TIMEOUT_SECONDS)
                
                if response.status_code == 200:
                    answers = response.json()
                    
                    st.success("✅ Processing complete!")
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Display answers
                    for question, answer in answers.items():
                        with st.expander(f"❓ {question}", expanded=True):
                            st.markdown(f"**Answer:**\n\n{answer}")
                    
                    # Download button
                    st.markdown("---")
                    json_str = json.dumps(answers, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download Answers as JSON",
                        data=json_str,
                        file_name="answers.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Error: {error_detail}")
                    st.code(f"Status Code: {response.status_code}\nDetail: {error_detail}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to API at {api_url}")
                st.info("💡 Make sure the FastAPI server is running:\n```bash\nuvicorn src.app:app --reload\n```")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The document might be too large or processing is taking too long.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, FastAPI, and LangChain")

