from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load precomputed ensemble predictions
ensemble_predictions = pd.read_csv('ensemble_predictions.csv', index_col='time', parse_dates=True)

# Define a route to get predictions for a specific date
@app.route('/predict', methods=['GET'])
def predict():
    # Get the date from the request (expected in format 'YYYY-MM-DD')
    date = request.args.get('date')
    
    # Convert the date to a datetime object
    try:
        date = pd.to_datetime(date)
    except Exception as e:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400

    # Check if the date is in the dataset
    if date in ensemble_predictions.index:
        prediction = ensemble_predictions.loc[date, 'prediction']
        return jsonify({'date': str(date), 'prediction': prediction})
    else:
        return jsonify({'error': 'Prediction for this date not available.'}), 404

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
