from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipe_line.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/CO2_Pre', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', results=None)
    else:
        try:
            data = CustomData(
                CO2=float(request.form.get('CO2')),
                CO2_1=float(request.form.get('CO2_1')),
                CO2_2=float(request.form.get('CO2_2')),
                CO2_3=float(request.form.get('CO2_3')),
                CO2_4=float(request.form.get('CO2_4'))
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            print(results)

            # Format results for HTML
            formatted_results = {
                'target_1': results['target_1'][0],
                'target_2': results['target_2'][0],
                'target_3': results['target_3'][0]
            }

            return render_template('home.html', results=formatted_results)

        except Exception as e:
            return render_template('home.html', results={'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)