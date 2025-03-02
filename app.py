import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
from pdf2image import convert_from_bytes
import pytesseract
from io import BytesIO
import config  # Ensure config.py exists with API keys & paths

# Configure Google Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

def extract_text_from_invoice(pdf_bytes):
    """Convert uploaded PDF invoice to text using OCR."""
    images = convert_from_bytes(pdf_bytes, poppler_path=config.POPPLER_PATH)  # Ensure poppler path is set
    extracted_text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return extracted_text

def extract_invoice_details(text):
    """Use Gemini AI to extract structured invoice details."""
    model = genai.GenerativeModel("gemini-1.5-pro")  # Ensure correct model

    try:
        response = model.generate_content(f"""
        Extract the following details from the given invoice text.
        Return ONLY CSV values (no headers, no extra text).
        Format: Total Amount,GST Number,Customer Name,Business Name,Date

        Invoice Text:
        {text}
        """)

        extracted_data = response.text.strip()
        return extracted_data  # Return CSV row

    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("📄 Invoice Extraction using Gemini AI")
st.write("Upload invoices and extract details automatically.")

uploaded_files = st.file_uploader("Upload Invoice PDFs", type=["pdf"], accept_multiple_files=True)

all_texts = []  # Store extracted text for global AI queries
csv_data = "File Name,Total Amount,GST Number,Customer Name,Business Name,Date\n"  # CSV Header
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.read()

        # Extract raw text
        extracted_text = extract_text_from_invoice(pdf_bytes)
        all_texts.append(f"Invoice: {uploaded_file.name}\n{extracted_text}")

        # Display extracted text
        st.subheader(f"Extracted Text - {uploaded_file.name}")
        st.text_area(f"Text from {uploaded_file.name}", extracted_text, height=200)

        # Extract structured invoice details
        extracted_details = extract_invoice_details(extracted_text)

        # Display structured invoice details
        if "Error" not in extracted_details:
            details = extracted_details.split(',')
            if len(details) == 5:
                st.write(f"**Total Amount:** {details[0]}")
                st.write(f"**GST Number:** {details[1]}")
                st.write(f"**Customer Name:** {details[2]}")
                st.write(f"**Business Name:** {details[3]}")
                st.write(f"**Date:** {details[4]}")

                csv_row = f"{uploaded_file.name},{extracted_details}\n"
                results.append(csv_row)
            else:
                st.error("Error parsing AI response. Check API output.")
                csv_row = f"{uploaded_file.name},Error Extracting Details\n"
                results.append(csv_row)
        else:
            st.error("Error extracting details from invoice.")
            csv_row = f"{uploaded_file.name},Error Extracting Details\n"
            results.append(csv_row)

        # "Ask Gemini AI" Button for Each Invoice
        with st.expander(f"🔍 Ask Gemini AI about {uploaded_file.name}"):
            user_query = st.text_input(f"Ask Gemini AI about {uploaded_file.name} (e.g., 'Summarize this invoice')")
            if st.button(f"Ask AI for {uploaded_file.name}"):
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(f"{user_query}\n\nInvoice Text:\n{extracted_text}")
                st.write("**AI Response:**")
                st.text_area("Gemini AI's Answer:", response.text, height=200)

    # Prepare CSV data for download
    csv_data += "".join(results)
    
    # Provide CSV & Text download buttons
    st.download_button("📥 Download CSV", csv_data, "invoice_results.csv", "text/csv")
    st.download_button("📥 Download Extracted Text", "\n".join(results), "invoice_results.txt", "text/plain")

    # 🌍 Global AI Query - Ask Gemini about ALL invoices
    st.subheader("🌍 Ask Gemini AI about ALL uploaded invoices")
    global_query = st.text_input("Ask anything about all invoices)")
    
    if st.button("🔍 Ask ME for All Invoices"):
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(f"{global_query}\n\nAll Invoices Text:\n{'\n\n'.join(all_texts)}")
        st.write("**AI Response:**")
        st.text_area("Gemini AI's Answer:", response.text, height=200)