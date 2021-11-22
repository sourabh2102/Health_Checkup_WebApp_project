from flask import Flask,render_template, url_for ,flash , redirect
import joblib
import requests
from flask import request
import numpy as np
import tensorflow

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')



# RELATED TO THE SQL DATABASE
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'


model = load_model('model111.h5')
model222=load_model("my_model.h5")

#FOR THE FIRST MODEL

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted
#FOR THE SECOND MODEL
def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model222.predict(data)
    return predicted


# home page
#@app.route('/')
#def home():
 #  return render_template('index.html')
 
 # Model1 for malaria
 #model11 for Pnuemonia

# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)

            predicted_class = np.asscalar(np.argmax(result, axis=1))  # ??
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))

@app.route('/upload11', methods=['POST','GET'])
def upload11_file():

    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            if(result>50):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 100-result
            return render_template('predict1.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)






#//////////////////////////////////////////////

#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

#db=SQLAlchemy(app)

#class User(db.Model):
##   username = db.Column(db.String(20), unique=True, nullable=False)
 #   email = db.Column(db.String(120), unique=True, nullable=False)
    #image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
 #   password = db.Column(db.String(60), nullable=False)
    #posts = db.relationship('Post', backref='author', lazy=True)

    #def __repr__(self):
    #   return f"User('{self.username}', '{self.email}', '{self.image_file}')"

@app.route("/")

@app.route("/home")
def home():
    return render_template("home.html")
 


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/cancer")
def cancer():
    return render_template("cancer.html")


@app.route("/diabetes")
def diabetes():
    #if form.validate_on_submit():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")


@app.route("/liver")
def liver():
    #if form.validate_on_submit():
    return render_template("liver.html")

@app.route("/kidney")
def kidney():
    #if form.validate_on_submit():
    return render_template("kidney.html")

@app.route("/Malaria")
def Malaria():
    return render_template("index.html")

@app.route("/Pneumonia")
def Pneumonia():
    return render_template("index2.html")

@app.route("/gethelp" ,methods=['GET'])
def gethelp():
    return render_template("map_form.html")
   
@app.route("/get_location", methods=['GET','POST'])
def get_location():
    latitude =  request.form['Latitude']
    longitude = request.form['Longitude']
    URL = "https://discover.search.hereapi.com/v1/discover"
    api_key = 'bKeiKFuiebb2wfx74iTbHWdpH0lr1xqPXH5qfhfVbJ8' # Acquire from developer.here.com
    query = 'hospital'
    limit = 5

    PARAMS = {
            'apikey':api_key,
            'q':query,
            'limit': limit,
            'at':'{},{}'.format(latitude,longitude)
         }
    # sending get request and saving the response as response object 
    r = requests.get(url = URL, params = PARAMS) 
    data = r.json()
    
    hospitalOne = data['items'][0]['title']
    hospitalOneDist = data['items'][0]['distance']
    hospitalOne_address =  data['items'][0]['address']['label']
    hospitalOne_latitude = data['items'][0]['position']['lat']
    hospitalOne_longitude = data['items'][0]['position']['lng']


    hospitalTwo = data['items'][1]['title']
    hospitalTwoDist = data['items'][1]['distance']
    hospitalTwo_address =  data['items'][1]['address']['label']
    hospitalTwo_latitude = data['items'][1]['position']['lat']
    hospitalTwo_longitude = data['items'][1]['position']['lng']

    hospitalThree = data['items'][2]['title']
    hospitalThreeDist = data['items'][2]['distance']
    hospitalThree_address =  data['items'][2]['address']['label']
    hospitalThree_latitude = data['items'][2]['position']['lat']
    hospitalThree_longitude = data['items'][2]['position']['lng']


    hospitalFour = data['items'][3]['title']
    hospitalFourDist = data['items'][3]['distance']
    hospitalFour_address =  data['items'][3]['address']['label']
    hospitalFour_latitude = data['items'][3]['position']['lat']
    hospitalFour_longitude = data['items'][3]['position']['lng']

    hospitalFive = data['items'][4]['title']
    hospitalFiveDist = data['items'][4]['distance']
    hospitalFive_address =  data['items'][4]['address']['label']
    hospitalFive_latitude = data['items'][4]['position']['lat']
    hospitalFive_longitude = data['items'][4]['position']['lng']
    
    
    return render_template('map.html',
                            latitude = latitude,
                            longitude = longitude,
                            apikey=api_key,
                            oneName=hospitalOne,
                            oneDistance=hospitalOneDist,
                            oneAddress=hospitalOne_address,
                            oneLatitude=hospitalOne_latitude,
                            oneLongitude=hospitalOne_longitude,
                            twoName=hospitalTwo,
                            twoDistance=hospitalTwoDist,
                            twoAddress=hospitalTwo_address,
                            twoLatitude=hospitalTwo_latitude,
                            twoLongitude=hospitalTwo_longitude,
                            threeName=hospitalThree,
                            threeDistance=hospitalThreeDist,
                            threeAddress=hospitalThree_address,
                            threeLatitude=hospitalThree_latitude,
                            threeLongitude=hospitalThree_longitude,
                            fourName=hospitalFour,
                            fourDistance=hospitalFourDist,                       fourAddress=hospitalFour_address,
                            fourLatitude=hospitalFour_latitude,
                            fourLongitude=hospitalFour_longitude,
                            fiveName=hospitalFive,
                            fiveDistance=hospitalFiveDist,                       fiveAddress=hospitalFive_address,
                            fiveLatitude=hospitalFive_latitude,
                            fiveLongitude=hospitalFive_longitude
                            )
    




def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = joblib.load("model1")
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load("model")
        result = loaded_model.predict(to_predict)
    elif(size==12):#Kidney
        loaded_model = joblib.load("model3")
        result = loaded_model.predict(to_predict)
    elif(size==10):
        loaded_model = joblib.load("model4")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load("model2")
        result =loaded_model.predict(to_predict)
    return result[0]

@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):
            result = ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):
            result = ValuePredictor(to_predict_list,11)
            #if int(result)==1:
            #   prediction ='diabetes'
            #else:
            #   prediction='Healthy' 
        elif(len(to_predict_list)==10):
            result = ValuePredictor(to_predict_list,10)
    if(int(result)==1):
        prediction='Sorry ! Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("result.html", prediction=prediction))


if __name__ == "__main__":
    app.run(debug=True)
