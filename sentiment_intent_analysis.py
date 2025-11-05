from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os
load_dotenv()

"""
Here we use the brettclaus/Hospital_Reviews model to classify the sentiment
"""
#This code is for predicting the intent
sentiment = pipeline("sentiment-analysis", model="brettclaus/Hospital_Reviews")
result = sentiment("I am very healthy")
print(result[0]['label'])



gemini_api_key = os.getenv("API_KEY")
genai.configure(api_key=gemini_api_key)

def build_intent_prompt(patient_utterance):
    return (
        "Given the patient's statement below, classify their intent as one of: "
        "'Seeking reassurance', 'Reporting symptoms', 'Expressing concern'.\n\n"
        f"Patient statement: \"{patient_utterance}\"\n\n"
        "Intent:"
    )

model = genai.GenerativeModel("gemini-2.5-flash")

utterance = "I'm a bit worried about my back pain, but I hope it gets better soon."
prompt = build_intent_prompt(utterance)

response = model.generate_content(prompt)

print(response.text)

"""
### How would you fine-tune BERT for medical sentiment detection?

We can fine-tune a **BERT model** by collecting medical text data along with their corresponding sentiment labels (e.g., *Anxious*, *Neutral*, *Reassured*). The labeled data is then used to fine-tune the pre-trained BERT model on this specific task, allowing it to learn sentiment patterns relevant to medical conversations.

---

### What datasets would you use for training a healthcare-specific sentiment model?

We can use **custom datasets** containing medical conversations and their corresponding sentiment labels.
Alternatively, we can explore **open-source medical sentiment datasets** or **healthcare-related text datasets** available on platforms like **Hugging Face Datasets** or **Kaggle** that include doctor–patient interactions, clinical notes, or patient feedback annotated for sentiment.
"""
