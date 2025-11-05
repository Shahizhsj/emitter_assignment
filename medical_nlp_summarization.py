import re
from transformers import pipeline
import spacy
from transformers import pipeline
from dotenv import load_dotenv
import os
load_dotenv()
nlp = spacy.load("en_core_web_sm")
"""
This is the description of the preprocess_text function

Here, we preprocess the text to clean and standardize it by performing the following steps:

Remove special characters Removes symbols such as *, >, [, ], _, `, and #.

Remove speaker tags Eliminates words like “Doctor:”, “Physician:”, and “Patient:” from the text.

Replace multiple newlines Converts multiple newline characters (\n) into a single space.

Condense extra spaces Reduces multiple spaces between words into a single space.

Convert text to lowercase Ensures all text is in lowercase for uniformity.
"""

def preprocess_text(text):
    text = re.sub(r'[*>\[\]_`#]', '', text)
    text = re.sub(r'\b(Doctor|Physician|Patient)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text
#Here we import our transcript
"""
Here we use the d4data/biomedical-ner-all model to extract the symptons
"""
text = os.getenv("trascript")
text=preprocess_text(text)
ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
results = ner(text)
print(results)
symptoms = [r["word"] for r in results if r["entity_group"].lower() in ["sign_symptom", "symptom"]]
treatment = [r["word"] for r in results if r["entity_group"].lower() in ["therapeutic_procedure"]]
# This is for the extracting the name 
doc = nlp(text)
name=[]
for ent in doc.ents:
    if ent.label_=="PERSON":
      name.append(ent.text)
# This is for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary=summarizer(text, max_length=130, min_length=30, do_sample=False)
output={"symptons":symptoms,"treatment":treatment,"name":name,"summary":summary[0]['summary_text']}
print(output)
"""
Sample Output
{'symptons': ['discomfort', 'pain', 'stiff', 'pain', 'anxiety', 'tenderness'],
 'treatment': ['painkillers', 'physiotherapy'],
 'name': ['jones', 'jones'],
 'summary': ' ms. jones was in a car accident last september. She suffered a whiplash injury and had to take painkillers for four weeks. She had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.'}
"""
