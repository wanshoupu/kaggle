from flask.ext.wtf import Form
from wtforms.fields import FileField, TextField, TextAreaField, SubmitField, BooleanField
 
class SubmitForm(Form):
  dataFile = FileField("Upload json file:")
  trainingMode = BooleanField("Cumulative training mode")
  trainingData = TextAreaField("Paste json data below:")
  submit = SubmitField("Submit")