import streamlit as st
import asyncio
import tempfile
import os
from pageindex.main import run_pipeline
from pageindex.utils import highlight_extracted_sections
import pymupdf
from PIL import Image
import io

# Check for required environment variables on startup
_required_checks = {
    "AZURE_OPENAI_ENDPOINT": ["AZURE_OPENAI_ENDPOINT"],
    "AZURE_OPENAI_DEPLOYMENT": ["AZURE_OPENAI_DEPLOYMENT"],
    "AZURE_OPENAI_API_KEY": ["AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"],
    "AZURE_DOC_INTEL_ENDPOINT": ["AZURE_DOC_INTEL_ENDPOINT", "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
    "AZURE_DOC_INTEL_KEY": ["AZURE_DOC_INTEL_KEY", "AZURE_DOCUMENT_INTELLIGENCE_KEY"],
}

_missing_env_vars = []
for display_name, env_names in _required_checks.items():
    if not any(os.getenv(var) for var in env_names):
        _missing_env_vars.append(display_name)

st.set_page_config(
    page_title="PageIndex - PDF Section Extractor",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .highlight-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def display_pdf_viewer(pdf_path):
    """Display PDF with page navigation using pymupdf"""
    try:
        doc = pymupdf.open(pdf_path)
        num_pages = len(doc)
        
        # Initialize session state for PDF page tracking
        if f"pdf_page_{pdf_path}" not in st.session_state:
            st.session_state[f"pdf_page_{pdf_path}"] = 0
        
        current_page = st.session_state[f"pdf_page_{pdf_path}"]
        
        # Page navigation controls
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
        
        with col1:
            if st.button("⬅️ Previous", key=f"prev_page_{pdf_path}"):
                if current_page > 0:
                    st.session_state[f"pdf_page_{pdf_path}"] -= 1
                    st.rerun()
        
        with col2:
            if st.button("Next ➡️", key=f"next_page_{pdf_path}"):
                if current_page < num_pages - 1:
                    st.session_state[f"pdf_page_{pdf_path}"] += 1
                    st.rerun()
        
        with col3:
            page_num = st.number_input(
                "Page",
                min_value=1,
                max_value=num_pages,
                value=current_page + 1,
                key=f"page_input_{pdf_path}"
            )
            if page_num - 1 != current_page:
                st.session_state[f"pdf_page_{pdf_path}"] = page_num - 1
                st.rerun()
        
        with col4:
            st.text(f"{current_page + 1} / {num_pages}")
        
        # Render the current page
        page = doc[current_page]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x zoom for better quality
        img = Image.open(io.BytesIO(pix.tobytes(output="png")))
        st.image(img, use_column_width=True)
        
        doc.close()
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">📄 PageIndex - PDF Section Extractor</h1>', unsafe_allow_html=True)
    st.markdown("Upload a PDF document and extract specific sections using AI-powered retrieval.")

    # Check for required environment variables
    if _missing_env_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(_missing_env_vars)}")
        st.markdown("""
        Please configure your `.env` file with:
        - `AZURE_OPENAI_ENDPOINT`
        - `AZURE_OPENAI_DEPLOYMENT`
        - `AZURE_OPENAI_API_KEY` (or `AZURE_OPENAI_KEY`)
        - `AZURE_DOC_INTEL_ENDPOINT`
        - `AZURE_DOC_INTEL_KEY`
        
        Then restart the app.
        """)
        return

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Display file info
        st.success(f"📁 File uploaded: {uploaded_file.name}")

        # Query input
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., Important Safety Information, Dosage Instructions, Side Effects",
            help="Describe the section you want to extract from the PDF"
        )

        # Process button
        if st.button("🔍 Extract Sections", type="primary", disabled=not query):
            if not query.strip():
                st.error("Please enter a search query")
                return

            with st.spinner("Processing PDF... This may take a few minutes"):
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        pdf_path = tmp_file.name

                    # Run the pipeline
                    sections = asyncio.run(run_pipeline(pdf_path, query))

                    # Generate highlighted PDF
                    highlighted_pdf_path = highlight_extracted_sections(pdf_path, sections)

                    # Clean up temp file
                    os.unlink(pdf_path)

                    # Store results in session state
                    st.session_state.sections = sections
                    st.session_state.highlighted_pdf = highlighted_pdf_path
                    st.session_state.query = query

                    st.success("✅ Processing complete!")

                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error processing PDF: {error_msg}")
                    
                    # Provide helpful diagnostics
                    with st.expander("📋 Troubleshooting Information"):
                        st.code(f"Error type: {type(e).__name__}\nMessage: {error_msg}", language="text")
                        st.markdown("""
                        **Common issues:**
                        - Ensure `.env` file is configured with Azure credentials
                        - Check that `AZURE_OPENAI_DEPLOYMENT` is set correctly
                        - Verify PDF file is valid and not corrupted
                        - Check network connectivity to Azure services
                        """)
                    return

        # Display results if available
        if 'sections' in st.session_state and st.session_state.sections:
            st.markdown("---")
            st.markdown("## 📋 Results & Verification")

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sections Found", len(st.session_state.sections))
            with col2:
                st.metric("Query", st.session_state.query)
            with col3:
                if 'highlighted_pdf' in st.session_state:
                    st.markdown('<div class="highlight-info">✅ PDF Highlighted</div>', unsafe_allow_html=True)

            # Display sections and PDF side-by-side
            col_sections, col_pdf = st.columns([1, 1.2], gap="large")

            with col_sections:
                st.markdown("### 📄 Extracted Sections")
                
                # Display each section
                for i, section in enumerate(st.session_state.sections, 1):
                    with st.expander(f"Section {i}: {section['title']}", expanded=i==1):
                        st.markdown("**Content:**")
                        st.markdown(section['text'])
                        
                        if st.button("📋 Copy", key=f"copy_{i}", use_container_width=True):
                            st.code(section['text'], language="markdown")
                            st.success("Content ready to copy!")

            with col_pdf:
                st.markdown("### 🎨 Highlighted PDF")
                
                # Display highlighted PDF
                if 'highlighted_pdf' in st.session_state and os.path.exists(st.session_state.highlighted_pdf):
                    display_pdf_viewer(st.session_state.highlighted_pdf)
                    
                    # Download button below PDF viewer
                    with open(st.session_state.highlighted_pdf, 'rb') as pdf_file:
                        st.download_button(
                            label="📥 Download Highlighted PDF",
                            data=pdf_file,
                            file_name=os.path.basename(st.session_state.highlighted_pdf),
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.info("✨ Yellow highlights show the extracted section locations in the original PDF!")
                else:
                    st.warning("PDF viewer not available")

        elif 'sections' in st.session_state and not st.session_state.sections:
            st.warning("⚠️ No sections found matching your query. Try a different search term.")

    else:
        # Instructions when no file is uploaded
        st.markdown("### 🚀 How to use:")
        st.markdown("1. **Upload a PDF** using the file uploader above")
        st.markdown("2. **Enter a search query** describing the section you want to extract")
        st.markdown("3. **Click 'Extract Sections'** to process the document")
        st.markdown("4. **Review results** and download the highlighted PDF for verification")

        st.markdown("### 💡 Tips:")
        st.markdown("- Be specific in your query (e.g., 'Important Safety Information' instead of 'safety')")
        st.markdown("- The system uses AI to find relevant sections across the entire document")
        st.markdown("- You'll get both the extracted text and a highlighted PDF showing where it was found")

if __name__ == "__main__":
    main()