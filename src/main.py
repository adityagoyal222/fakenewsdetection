from fastapi import FastAPI, HTTPException, Query
from classifier import predict_news

app = FastAPI(title="Game Recommender")

# simple get request to predict the classification of an article
@app.get("/predict")
def predict(text: str = Query(..., description="Enter the text for prediction")):
    prediction, confidence = predict_news(text)
    print(prediction)
    return {"label": prediction, "confidence": float(confidence)}