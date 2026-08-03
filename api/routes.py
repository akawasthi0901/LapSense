from fastapi import APIRouter, HTTPException

from .schemas import HealthResponse, LaptopInput, PredictionResponse
from .utils import load_model_bundle, prepare_features


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
	return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictionResponse)
def predict_price(laptop: LaptopInput) -> PredictionResponse:
	try:
		bundle = load_model_bundle()
		features = prepare_features(laptop, bundle["features"])
		prediction = float(bundle["model"].predict(features)[0])
		return PredictionResponse(
			predicted_price_euros=round(prediction, 2),
			model_name=bundle.get("model_name", "ML Model"),
		)
	except FileNotFoundError as error:
		raise HTTPException(status_code=503, detail=str(error)) from error
	except Exception as error:
		raise HTTPException(status_code=500, detail="Prediction failed") from error
