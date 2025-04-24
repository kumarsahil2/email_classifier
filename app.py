from models import train_model

if __name__ == "__main__":
    train_model("data/combined_emails_with_natural_pii.csv")
    print("✅ Model trained and saved.")
