from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method== 'GET':
        return render_template('index.html')
    else:
        data=CustomData(
            Age=int(request.form.get('Age')),
            Gender=request.form.get('Gender'),
            Daily_Screen_Time_hrs=float(request.form.get('Daily_Screen_Time(hrs)')),
            Sleep_Quality=int(request.form.get('Sleep_Quality(1-10)')),
            Stress_Level=int(request.form.get('Stress_Level(1-10)')),
            Days_Without_Social_Media=int(request.form.get('Days_Without_Social_Media')),
            Exercise_Frequency_week=int(request.form.get('Exercise_Frequency(week)')),
            Social_Media_Platform=request.form.get('Social_Media_Platform')
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        # Before Prediction

        # Run prediction
        predict_pipeline = PredictPipeline()
        # Mid Prediction
        results = predict_pipeline.predict(pred_df)
        # After Prediction

        return render_template('index.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)