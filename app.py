from flask import Flask, render_template, request, flash
from forms import SubmitForm
import logging
from logging.handlers import RotatingFileHandler
from training import Trainer
from analyzer import parseJson,plot,groupByHour
from datetime import datetime as dt
from datetime import timedelta as td

RESOURCES_DIR = 'resources'

import os
if not os.path.exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)

app = Flask(__name__, static_url_path = '', static_folder = RESOURCES_DIR)
app.secret_key = 'development key'  
app.config['RESOURCES_DIR'] = RESOURCES_DIR 

trainer = Trainer(app)

def makePlot(timestamps, predResult):
    import time
    imgname = time.strftime("%Y%m%d-%H%M%S")
    img = imgname+'.png'
    plt = plot([
        {'data' : [timestamps,predResult], 'xlabel' : 'Hour', 'ylabel':'Request per hour', 'title':'Predictions of demand level'},
        ])
    plt.savefig('resources/'+img, bbox_inches='tight')
    return img

@app.route('/')
def home():
  return render_template('home.html')
  
@app.route('/predict')
def predict():
  dateGrid = [dt(2012, 5, 1) + td(hours=x) for x in range(0, 15 * 24)]
  dateStr = map(lambda t:t.strftime("%Y-%m-%dT%H:%M:%S%Z"),dateGrid)
  predResult = trainer.predict(dateGrid)
  predStr = map(lambda x : '{:.2f}'.format(x), predResult)
  predictions= zip(dateStr, predStr)
  img=makePlot(dateGrid,predResult)
  return render_template('predict.html', predictions=predictions, img=img)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
  global trainer
  form = SubmitForm()
 
  if request.method == 'POST':
    if form.validate() == False:
      flash('All fields are required.')
      return render_template('submit.html', form=form)
    else:
      trainingdata = ''
      if form.dataFile.data :
        trainingdata += request.files[form.dataFile.name].read()
      if form.trainingData.data :
        trainingdata += form.trainingData.data

      if trainingdata :
        timestampes = parseJson(trainingdata)
        img1,img2,metrics = trainer.train(timestampes)

      return render_template('submit.html', success=True, img1=img1, img2=img2, metrics=metrics)
 
  elif request.method == 'GET':
    return render_template('submit.html', form=form)

if __name__ == '__main__':
  handler = RotatingFileHandler('/tmp/foo.log', maxBytes=10000, backupCount=1)
  handler.setLevel(logging.INFO)
  app.logger.addHandler(handler)

  app.run(debug=True)