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
text="""Physician: Good morning, Ms. Jones. How are you feeling today?

Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.

Physician: I understand you were in a car accident last September. Can you walk me through what happened?

Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.

Physician: That sounds like a strong impact. Were you wearing your seatbelt?

Patient: Yes, I always do.

Physician: What did you feel immediately after the accident?

Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.

Physician: Did you seek medical attention at that time?

Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.

Physician: How did things progress after that?

Patient: The first four weeks were rough. My neck and backpain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.

Physician: That makes sense. Are you still experiencing pain now?

Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.

Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?

Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.

Physician: And how has this impacted your daily life? Work, hobbies, anything like that?

Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.

Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.

[Physical Examination Conducted]

Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.

Patient: That’s a relief!

Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.

Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?

Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.

Patient: Thank you, doctor. I appreciate it.

Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.

"""
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
