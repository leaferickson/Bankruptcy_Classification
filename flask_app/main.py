from flask import Flask, request, render_template
from py_muffins_cupcakes import bankruptcy_prediction
import pandas as pd

# create a flask object
app = Flask(__name__)

# creates an association between the / page and the entry_page function (defaults to GET)
@app.route('/')
def entry_page():
    return render_template('index.html')

# creates an association between the /predict_recipe page and the render_message function
# (includes POST requests which allow users to enter in data via form)
@app.route('/predict_bankruptcy/', methods=['GET', 'POST'])
def render_message(data = upload_data):

    # show user final message
    # final_message = bankruptcy_prediction()
    return render_template('/templates/test.html')

if __name__ == '__main__':
    app.run(debug=True)