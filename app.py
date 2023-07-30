# API server using Flask
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from util import * # contains helper functions


#loading the models 
import joblib
model1 = joblib.load('svm_model.pkl')
model2 = joblib.load('knn_model.pkl')
model3 = joblib.load('LR_model.pkl')
#tfidf vectorizer and transformer
vectorizer=joblib.load('tfidf_vectorizer.pkl')
model = SentenceTransformer('all-MiniLM-L6-v2')


# Initialize the Flask app
app = Flask(__name__)

# Define the API route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/<selected_model>', methods=['POST'])
def predict(selected_model):
    try:
        # Get the input text from the request
        data = request.get_json(force=True)
        text = data['text']
        #selected_model = data['selected_model']
        #calling the  functions for text pre processing
        text= spacy_tokenizer(text)

        # Transform the user input into its TF-IDF representation for svm knn
        input_tfidf = vectorizer.transform([text])

        # Transform the user input into its embedding for lR
        text= spacy_tokenizer(text)
        text= model.encode(text).reshape(1,-1)

        # Make the prediction using the ML model
            
        if selected_model == 'model1':
            prediction = model1.predict(input_tfidf)[0]
        elif selected_model == 'model2':
            prediction = model2.predict(input_tfidf)[0]
        elif selected_model == 'model3':
            dic= get_dict()
            i=model3.predict(text)[0]#(op is from 0-40 unique labels)
            prediction = dic[i]#decode and get the disease
        else:
            return jsonify({'error': 'Invalid model selection.'})

        # Return the prediction as a JSON response
        response = {
            'prediction': prediction
        }
        
        return jsonify(response)

    except Exception as e:
        # Handle any errors and return an error message
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app on localhost with port 5000
    app.run(host='0.0.0.0', port=5000)
