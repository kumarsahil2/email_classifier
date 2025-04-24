📧 Email Classification for Support Teams
This project solves a real-world challenge of classifying support emails while protecting user privacy by masking Personally Identifiable Information (PII) using strict regular expressions — without LLMs. The email is then passed to a trained ML model (Naive Bayes) to categorize the message into appropriate support types.

🚀 Features
🔒 PII Masking:

Detects and masks emails, phone numbers, Aadhar, credit cards, CVV, expiry, DoB, full names.

Handles edge cases like "+91" prefix, hyphens, spaces, and more.

🧠 Email Classification:

Trained using Naive Bayes on TF-IDF features.

Categories emails into support-related types.

⚡ API Endpoint:

FastAPI-powered POST API that returns:

Masked email body

Extracted PII entities

Predicted support category

📁 Project Structure
graphql
Copy
Edit
.
├── api.py # FastAPI for prediction API
├── app.py # Entry point for training the model
├── models.py # Model training pipeline
├── utils.py # PII masking logic
├── email_classifier.pkl # Saved trained model
├── data/
│ └── combined_emails_with_natural_pii.csv
└── README.md
🧪 Sample Input & Output
POST /classify_email
Input

json
Copy
Edit
{
"input_email_body": "Hello, my name is Rahul Sharma. My card number is 1234-5678-9876-5432 and CVV: 123. Kindly help."
}
Output

json
Copy
Edit
{
"input_email_body": "Hello, my name is Rahul Sharma. My card number is 1234-5678-9876-5432 and CVV: 123. Kindly help.",
"list_of_masked_entities": [
{
"position": [18, 41],
"classification": "full_name",
"entity": "Rahul Sharma"
},
{
"position": [61, 96],
"classification": "credit_debit_no",
"entity": "1234-5678-9876-5432"
},
{
"position": [101, 110],
"classification": "cvv_no",
"entity": "CVV: 123"
}
],
"masked_email": "Hello, my name is [full_name]. My card number is [credit_debit_no] and [cvv_no]. Kindly help.",
"category_of_the_email": "problem"
}
⚙️ How to Run Locally

1. 🔧 Install Requirements
   bash
   Copy
   Edit
   pip install -r requirements.txt
2. 🧠 Train the Model
   bash
   Copy
   Edit
   python app.py
   This will save email_classifier.pkl after training.

3. 🚀 Start the API
   bash
   Copy
   Edit
   uvicorn api:app --reload --port 7860
   POST requests can be sent to http://localhost:7860/classify_email

🧰 Requirements
nginx
Copy
Edit
fastapi
uvicorn
scikit-learn
pandas
nltk
Install via:

bash
Copy
Edit
pip install fastapi uvicorn scikit-learn pandas nltk
☁️ Deployment on Hugging Face Spaces
To deploy:

Use api.py as the main API

Make sure email_classifier.pkl is uploaded

Set api:app as the entrypoint in Spaces configuration

Space type: "Gradio / FastAPI"
