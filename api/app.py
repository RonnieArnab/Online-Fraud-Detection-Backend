from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pickle
import numpy as np



app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://online-fraud-detection-frontend.vercel.app/"}})

# Load the trained machine learning modelpip
model = pickle.load(open('./models/model.pkl', 'rb'))

@app.route("/api/members", methods=['POST', 'OPTIONS'])
def members():
    if request.method == 'OPTIONS':
        # Respond to the OPTIONS request
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'https://online-fraud-detection-frontend.vercel.app/')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        # Get the input data from the request
        input_data = request.get_json()

        # Assuming the input data has the same keys as the model expects
        # Make sure to adjust this part based on your actual model's input features
        input_features = [
            int(input_data["type"]),
            int(input_data["amount"]),
            int(input_data["oldbalanceOrg"]),
            int(input_data["newbalanceOrig"])
        ]

        print("Input Data:", input_data)

        # Convert input features to a NumPy array and reshape
        input_features = np.array(input_features).reshape(1, -1)


        # Make predictions using the loaded model
        prediction = model.predict(input_features)

        print(prediction)


        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        # Handle exceptions and return an error response if needed
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
