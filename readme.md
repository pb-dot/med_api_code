First Go to the helpers folder to run the gen1.py and gen2.py to generate the model.pkl files
ie SVM model and KNN model with tfidf text representation in gen1.py
and LR model which uses transformer as text representation in gen2.py

The util.py has hepler function needed by app.py for prediction and text preprocessing

While deploying remove the helpers folder as we already have .pkl files

### Run the whole application using

pip install -r requirements.txt --user <br>
cd helpers folder<br>
python gen1.py<br>
python gen2.py<br>
Move all the .pkl and .csv file outside helpers folder<br>
cd ..<br>
streamlit run app.py<br>
