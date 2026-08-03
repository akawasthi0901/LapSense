from pydantic import BaseModel, Field


class LaptopInput(BaseModel):
	"""Laptop specifications accepted by the prediction endpoint."""

	company: str = Field(..., examples=["Dell"])
	type_name: str = Field(..., examples=["Notebook"])
	os: str = Field(..., examples=["Windows"])
	ram: int = Field(..., gt=0, le=128, examples=[8])
	cpu_brand: str = Field(..., examples=["Intel"])
	gpu_brand: str = Field(..., examples=["Intel"])
	weight: float = Field(..., gt=0, le=10, examples=[1.8])
	hdd: int = Field(0, ge=0, le=1, examples=[0])
	ssd: int = Field(1, ge=0, le=1, examples=[1])
	ppi: float = Field(..., gt=0, examples=[141.0])
	touchscreen: int = Field(0, ge=0, le=1, examples=[0])
	ips: int = Field(1, ge=0, le=1, examples=[1])


class PredictionResponse(BaseModel):
	predicted_price_euros: float
	model_name: str


class HealthResponse(BaseModel):
	status: str
