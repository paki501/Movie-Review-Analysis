
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import string
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ✅ Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Load the vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model and vectorizer loaded successfully!")

# ✅ Initialize FastAPI app
app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Define input data model
class InputText(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(data: InputText):
    text = data.text  # Get input text
    vectorized_text = vectorizer.transform([text])  # Convert text to numerical vectors

    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    return {"text": text, "sentiment": sentiment}

@app.post("/predict")
async def predict_sentiment(request: InputText):
    try:
        # Vectorize input text
        text_vectorized = vectorizer.transform([request.text])
        
        # Predict sentiment
        prediction = model.predict(text_vectorized)[0]
        
        # Convert prediction to readable format
        sentiment = "Positive" if prediction == 1 else "Negative"
        return {"sentiment": sentiment}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Run API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




from sklearn.utils.validation import check_is_fitted
import joblib

file_path = "model.pkl"

try:
    model = joblib.load(file_path)
    check_is_fitted(model)  # Check if model is trained
    print("Model is loaded and trained.")
except Exception as e:
    print("Error:", e)
