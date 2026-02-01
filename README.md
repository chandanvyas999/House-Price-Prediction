# House Price Prediction

A generic machine learning project to predict house prices based on features like bedrooms, bathrooms, size, and zip code.

## Project Structure

This project has been restructured for better organization:

- **`app/`**: Contains the Flask application logic.
  - `main.py`: The entry point for the web application.
  - `templates/`: HTML templates for the frontend.
- **`data/`**: Stores the datasets used for training and testing.
  - `final_dataset.csv`: The processed dataset used by the application.
  - `Bengaluru_House_Data.csv`, `ParisHousing.csv`, etc.: Raw datasets.
- **`models/`**: Contains the serialized machine learning models.
  - `RidgeModel.pkl`: The trained Ridge Regression model used for predictions.
- **`notebooks/`**: Jupyter notebooks for data exploration and model training.
  - `House Price Prediction.ipynb`: The notebook where the model was created.

## Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install flask pandas scikit-learn
    ```

## Usage

1.  Navigate to the `app` directory (or run from root):
    ```bash
    python app/main.py
    ```
2.  Open your browser and navigate to `http://127.0.0.1:5000`.
3.  Enter the house details (bedrooms, bathrooms, size, zip code).
4.  Click "Predict Price" to see the estimated price.

## Development

- To retrain the model, check the `notebooks/` directory.
- To modify the web app, edit `app/main.py` or `app/templates/index.html`.
