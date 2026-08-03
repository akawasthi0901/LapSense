# Run Instructions for LapSense

This file explains how to run the full laptop price prediction project on Windows.

## 1. Open the project folder
In PowerShell, go to the project directory:

```powershell
cd C:\Users\akash\Desktop\Naukri\Projects\LapSense
```

## 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## 3. Install dependencies
```powershell
pip install -r requirements.txt
```

## 4. Make sure the dataset exists
The project expects the dataset here:

```text
data\laptop_price.csv
```

If the file is missing, place the CSV file into the data folder before training.

## 5. Train the model
Run the training pipeline:

```powershell
python laptopPricePredictorAPI.py
```

This will:
- load the dataset
- create EDA plots in the eda_plots folder
- train and compare models
- save the trained model to:

```text
model\laptop_price_model.pkl
```

## 6. Start the web app
After training completes, launch the Streamlit app:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 7. Use the app
Once the app opens in the browser:
- enter laptop details
- click Predict Price
- view the estimated price in EUR, INR, and USD

## Optional
To stop the app, press Ctrl + C in the terminal.

To leave the virtual environment later, run:

```powershell
deactivate
```
