from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import sys

app = Flask(__name__)

# Add robust logging to stdout
def log(msg):
    print(f"[DEBUG] {msg}", file=sys.stdout)
    sys.stdout.flush()

try:
    # Construct absolute paths to resources
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'final_dataset.csv')
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'RidgeModel.pkl')

    log(f"Attempting to load data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        log(f"CRITICAL ERROR: Data file not found at {DATA_PATH}")
        data = pd.DataFrame() # Empty dataframe to prevent crash on import
    else:
        data = pd.read_csv(DATA_PATH)
        log("Data loaded successfully.")
        
        # clean column names
        data.columns = data.columns.str.strip()
        log(f"Columns found: {data.columns.tolist()}")
        log(f"Data shape: {data.shape}")
        log(f"First row: {data.iloc[0].to_dict()}")

    log(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        log(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        pipe = None
    else:
        pipe = pickle.load(open(MODEL_PATH, 'rb'))
        log("Model loaded successfully.")

except Exception as e:
    log(f"CRITICAL ERROR during initialization: {str(e)}")
    data = pd.DataFrame()
    pipe = None

@app.route('/')
def index():
    log("Request received for index page")
    
    if data.empty:
        log("Data is empty. Returning placeholders.")
        return render_template('index.html', bedrooms=[], bathrooms=[], sizes=[], zip_codes=[])

    try:
        # Robust sorting with dropna to avoid NaN issues
        bedrooms = sorted(data['beds'].dropna().unique())
        bathrooms = sorted(data['baths'].dropna().unique())
        sizes = sorted(data['size'].dropna().unique())
        zip_codes = sorted(data['zip_code'].dropna().unique())

        log(f"Prepared options: {len(bedrooms)} beds, {len(bathrooms)} baths, {len(sizes)} sizes, {len(zip_codes)} zips")
    except Exception as e:
        log(f"Error preparing options: {e}")
        bedrooms, bathrooms, sizes, zip_codes = [], [], [], []

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    if pipe is None:
        return "Model not loaded properly."

    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    log(f"Prediction request: beds={bedrooms}, baths={bathrooms}, size={size}, zip={zipcode}")

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])
    
    # Handle unknown categories in the input data
    for column in input_data.columns:
        if column in data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                log(f"Unknown categories in {column}: {unknown_categories}")
                # Handle unknown categories (e.g., replace with a default value - mode)
                if not data[column].mode().empty:
                     input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    try:
        prediction = pipe.predict(input_data)[0]
    except Exception as e:
        log(f"Prediction error: {e}")
        return str(e)

    return str(prediction)

if __name__ == "__main__":
    log("Starting Flask server...")
    app.run(debug=True, port=5000)   
           