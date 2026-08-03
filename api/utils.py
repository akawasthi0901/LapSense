from pathlib import Path
from typing import List

import joblib
import pandas as pd

from .schemas import LaptopInput


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "laptop_price_model.pkl"


def load_model_bundle() -> dict:
	if not MODEL_PATH.exists():
		raise FileNotFoundError(
			f"Model not found at {MODEL_PATH}. Run laptopPricePredictorAPI.py first."
		)
	return joblib.load(MODEL_PATH)


def prepare_features(laptop: LaptopInput, feature_order: List[str]) -> pd.DataFrame:
	"""Convert the API schema to the engineered feature names used by the model."""
	values = {
		"Company": laptop.company,
		"TypeName": laptop.type_name,
		"Ram": laptop.ram,
		"Weight": laptop.weight,
		"Touchscreen": laptop.touchscreen,
		"Ips": laptop.ips,
		"ppi": laptop.ppi,
		"Cpu brand": laptop.cpu_brand,
		"HDD": laptop.hdd,
		"SSD": laptop.ssd,
		"Gpu brand": laptop.gpu_brand,
		"os": laptop.os,
	}
	return pd.DataFrame([[values[name] for name in feature_order]], columns=feature_order)
