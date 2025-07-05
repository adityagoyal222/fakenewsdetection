from fastapi import FastAPI, HTTPException, Query
from classifier import predict_news

app = FastAPI(title="Game Recommender")

@app.get("/predict")
def predict(text: str = Query(..., description="Enter the text for prediction")):
    print(text)
    prediction, confidence = predict_news(text)
    print(prediction)
    return {"label": prediction, "confidence": float(confidence)}