from fastapi import FastAPI

from api.routes import router


app = FastAPI(
	title="LapSense Laptop Price API",
	version="1.0.0",
	description="Predict laptop prices from hardware specifications.",
)

app.include_router(router, prefix="/api")


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("fastapiServer:app", host="127.0.0.1", port=8000, reload=True)
