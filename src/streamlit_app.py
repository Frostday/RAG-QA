"""Streamlit frontend for the Question-Answering Bot API."""
import json
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Question-Answering Bot",
    page_icon="🤖",
    layout="wide"
)

# Default API URL
api_url = "http://localhost:8000"

# Title and description
st.title("Question-Answering Bot")
st.markdown("Upload a document and questions to get AI-powered answers based on your document content.")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Document")
    document_file = st.file_uploader(
        "Choose document file",
        type=["pdf", "json"],
        help="Upload a PDF or JSON document"
    )
    
    if document_file:
        st.success(f"✅ {document_file.name} uploaded")
        st.caption(f"File size: {document_file.size / 1024:.2f} KB")

with col2:
    st.subheader("❓ Upload Questions")
    questions_file = st.file_uploader(
        "Choose questions file",
        type=["json"],
        help="Upload a JSON file containing questions"
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
            
            st.success(f"✅ {questions_file.name} uploaded")
            st.caption(f"Found {len(questions_list)} question(s)")
            
            # Show preview
            with st.expander("Preview Questions"):
                for i, q in enumerate(questions_list, 1):
                    st.markdown(f"{i}. {q}")
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
                response = requests.post(endpoint, files=files, timeout=300)
                
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
                st.info("💡 Make sure the FastAPI server is running:\n```bash\nuvicorn app:app --reload\n```")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The document might be too large or processing is taking too long.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.caption("Built with Streamlit, FastAPI, and LangChain")

