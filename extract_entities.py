import re
import json
import spacy
import google.generativeai as genai
from config import GEMINI_API_KEY  # Import API key

# Load spaCy NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def extract_entities(text):
    """Extracts structured data from invoice text."""

    # Regex patterns for invoice details
    invoice_number_pattern = r"(?:Invoice\s*No[:\s]*|Invoice#[:\s]*)\s*([\w-]+)"
    date_pattern = r"(?:Date[:\s]*)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    gstin_pattern = r"(?:GSTIN[:\s]*|GST No[:\s]*)\s*([A-Z0-9]{15})"
    total_amount_pattern = r"(?:Total\s*Amount[:\s]*|Grand Total[:\s]*|Amount Due[:\s]*)\s*\$?([\d,]+\.\d{2})"

    # Extract values using regex
    invoice_number = re.search(invoice_number_pattern, text)
    invoice_date = re.search(date_pattern, text)
    gstin = re.search(gstin_pattern, text)
    total_amount = re.search(total_amount_pattern, text)

    # Extract customer or restaurant name
    extracted_name = extract_names(text)

    # Convert to structured format
    structured_data = {
        "Invoice Number": invoice_number.group(1) if invoice_number else "Not Found",
        "Invoice Date": invoice_date.group(1) if invoice_date else "Not Found",
        "GSTIN": gstin.group(1) if gstin else "Not Found",
        "Total Amount": total_amount.group(1) if total_amount else "Not Found",
        "Customer/Restaurant Name": extracted_name if extracted_name else "Not Found"
    }

    # Send to Gemini AI for correction
    structured_data = enhance_with_gemini(text, structured_data)

    return structured_data

def extract_names(text):
    """Uses NLP to extract names from invoice text."""
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON"]]

    return names[0] if names else "Not Found"

def enhance_with_gemini(invoice_text, extracted_data):
    """Uses Gemini AI to verify and improve extracted invoice data."""
    prompt = f"""
    The following is an extracted invoice text:

    {invoice_text}

    The extracted details are:
    {json.dumps(extracted_data, indent=2)}

    Please verify if the extracted values are correct. If any values are missing or incorrect, provide the corrected version in the same JSON format.
    """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content([prompt])

    try:
        # Parse JSON response from Gemini AI
        corrected_data = json.loads(response.text.strip())
        return corrected_data
    except json.JSONDecodeError:
        return extracted_data  # Return original data if Gemini response is not valid JSON

# Testing the function
if __name__ == "__main__":
    sample_text = """
    Invoice No: Z24KAO0T000472206
    Date: 03/10/2023
    GSTIN: 22AAAAA0000A1Z5
    Customer: John Doe
    Restaurant: ABC Restaurant Pvt Ltd
    Grand Total: $1899.98
    """
    
    structured_output = extract_entities(sample_text)
    print(json.dumps(structured_output, indent=4))
