from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
from utils import mask_pii
import uvicorn

# Load classifier
with open("email_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI()

class EmailInput(BaseModel):
    input_email_body: str

@app.post("/classify_email")
async def classify_email(payload: EmailInput):
    raw_email = payload.input_email_body
    masked_email, pii_entities = mask_pii(raw_email)
    category = model.predict([masked_email])[0]

    return {
        "input_email_body": raw_email,
        "list_of_masked_entities": pii_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
