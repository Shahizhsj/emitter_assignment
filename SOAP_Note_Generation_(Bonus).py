import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

gemini_api_key = os.getenv("API_KEY")
genai.configure(api_key=gemini_api_key)


def build_soap_prompt(transcript):
    return (
        "You are a clinical assistant. Convert the following patient-doctor conversation into a SOAP note in JSON format. "
        "SOAP stands for Subjective, Objective, Assessment, and Plan.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Format:\n{\n  \"Subjective\": \"...\",\n  \"Objective\": \"...\",\n  \"Assessment\": \"...\",\n  \"Plan\": \"...\"\n}\n\n"
        "Output the result as valid JSON only, no explanation."
    )

transcript = '''Doctor: How are you feeling today?
Patient: I had a car accident. My neck and back hurt a lot for four weeks.
Doctor: Did you receive treatment?
Patient: Yes, I had ten physiotherapy sessions, and now I only have occasional back pain.'''

prompt = build_soap_prompt(transcript)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(prompt)

print(response.text)


"""raining an NLP Model for SOAP Mapping
Overview
Mapping transcribed medical text into SOAP format involves:

Annotating transcripts with SOAP labels (Subjective, Objective, Assessment, Plan).
Framing the task as sequence labeling, text segmentation, or multi-span extraction.
Fine-tuning transformer-based models (Bio_ClinicalBERT, GPT, Gemma, etc.) on annotated clinical dialogues to identify section boundaries, extract facts, and structure them into JSON or text format.
Enhancing performance with rule-based heuristics using keywords like “pain”, “treatment”, “plan”, “exam”, etc., as fallback cues.
Improving SOAP Note Generation Accuracy
Rule-Based Techniques
If using rule-based methods:

Create patterns for each section.
Use regular expressions and pattern matching to extract relevant information.
Deep Learning Techniques
If using deep learning methods:

Perform fine-tuning of the transformer model as discussed above.
"""
