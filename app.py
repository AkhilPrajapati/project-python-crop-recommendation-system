from flask import Flask
from flask import render_template
from flask import redirect
from flask import url_for
from flask import request
from flask import session
from flask import jsonify
from flask import abort
from flask_mysqldb import MySQL

import string
import pandas as pd
import numpy as np
import pickle
import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")



app = Flask(__name__)






loaded_modelRFC = pickle.load(open('modelRFC.pkl', 'rb'))
loaded_modelNB = pickle.load(open('modelNB.pkl', 'rb'))
loaded_modelDT = pickle.load(open('modelDT.pkl', 'rb'))
loaded_modelKNN = pickle.load(open('modelKNN.pkl', 'rb'))

accuracy_valueRFC = pickle.load(open('accuracy_modelRFC.pkl', 'rb'))
accuracy_valueNB = pickle.load(open('accuracy_modelNB.pkl', 'rb'))
accuracy_valueDT = pickle.load(open('accuracy_modelDT.pkl', 'rb'))
accuracy_valueKNN = pickle.load(open('accuracy_modelKNN.pkl', 'rb'))

loginWarning = str("Warning: Please Login First...")


# any name that is super secret key
app.secret_key = "super secret key"
# app.secret_key = 'login'


app.config['MYSQL_Host'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_project'
mysql = MySQL(app)


@app.errorhandler(404)
def page_not_found(e):
    print("Here is error")
    print(e)
    return "page not found"

@app.errorhandler(500)
def server_error(e):
    print("Here is error")
    print(e)
    return "server error"






# FEEDBACK ROUTING AND CONTROLLER.//////////////////////////////////

@app.route('/feedback')
def feedback():
    return render_template("startingtemplates/feedback.html")

@app.route('/feedback', methods = ["POST", "GET"])
def feedback_controller():
    varDate = datetime.datetime.now()
    year = varDate.year
    month = varDate.month
    day = varDate.day
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        crop = request.form['crop']
        message = request.form['message']
        date = str(year)+"-"+str(month)+"-"+str(day)
        myCursor = mysql.connection.cursor()
        myCursor.execute("insert into feedbacks(name, email, subject, crop, message, date) values(%s, %s, %s, %s, %s, %s)", (name, email, subject, crop, message, date))
        mysql.connection.commit()
        return render_template('startingtemplates/feedback.html')
    
@app.route('/admin/feedback/delete/<string:id>', methods = ['GET'])
def delete_feedback(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM feedbacks WHERE id=%s", (id,))
    # id paxi COMMA chutaunu bhyena
    mysql.connection.commit()
    print("data deleted in console")
    return redirect(url_for('admin_feedback'))


# UPDATE AND DELETE ROUTES.////////////////////////

@app.route("/update/<string:id>", methods = ['get'])
def update(id):
    myCursor = mysql.connection.cursor()
    myCursor.execute("select * from crops where id = %s", (id))
    data = myCursor.fetchall()
    myCursor.close()
    return render_template("entryforms/updatecrops.html", data = data)

@app.route('/update', methods=['POST'])
def update_controller():
    # secret key hunu parxa
    if request.method == 'POST':
        id = request.form['id']
        name = request.form['name']
        description = request.form['description']
        season = request.form["season"]
        scientific_name = request.form['scientific_name']
        image_url = request.form['image_url']
        print(id)
        print(name)
        print(description)
        print(season)
        print(scientific_name)
        print(image_url)
        myCursor = mysql.connection.cursor()

        myCursor.execute("UPDATE crops SET name=%s, description=%s, season=%s, scientific_name=%s, image_url=%s WHERE id=%s", (name, description, season, scientific_name, image_url, id))
        # flash("Data Updated Successfully")
        mysql.connection.commit()
        return redirect(url_for('admin_crops'))






@app.route('/admin/crops/delete/<string:id>', methods = ['GET'])
def delete(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM crops WHERE id=%s", (id,))
    # id paxi COMMA chutaunu bhyena
    mysql.connection.commit()
    print("data deleted in console")
    return redirect(url_for('admin_crops'))









# AUTNENTICATION ROUTES/////////////////////////////////

@app.route('/logout', methods = ["POST", "GET"])
def logout():
    session.pop('email', None)

    # session.pop('loggedin', None)
    # session.pop('username', None)
    # return "logged out"
    return redirect(url_for('login'))

@app.route("/login")
def login():
    return render_template("authtemplates/login.html")

@app.route('/login_controller', methods = ["POST", "GET"])
def login_controller():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        print(email)
        print(password)
        myCursor = mysql.connection.cursor()
        myCursor.execute("SELECT * FROM `admins` WHERE email = %s AND password = %s;", (email, password))
        record = myCursor.fetchone()
        print(record)
        print(email)
        print(password)
        print ("errr")
        if record:
            # session['loggedin'] = True
            # session['loggrdin']
            session['email'] = record[3]
            print(record[3])
            return redirect(url_for('dashboard'))
        else:
            msg = "please try again"
    else:
        return "error"
        # return render_template('index.html', msg = msg)


@app.route('/register')
def register():
    return render_template('authtemplates/register.html')

@app.route('/register_controller', methods = ["POST", "GET"])
def register_controller():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        password = request.form['password']
        myCursor = mysql.connection.cursor()
        myCursor.execute("insert into admins(fname, lname, email, password) values(%s, %s, %s, %s)", (fname, lname, email, password))
        mysql.connection.commit()

        myCursor.execute("SELECT * FROM `admins` WHERE email = %s AND password = %s;", (email, password))
        record = myCursor.fetchone()

        session['email'] = record[3]
        return render_template("admintemplates/dashboard.html")
    


###### STARTING TEMPLATE ROUTINGS ########



@app.route("/chat")
def chat_route():
    return render_template("startingtemplates/chat.html")

@app.route('/')
def index():
    # return render_template("startingtemplates/chat.html")
    myCursor = mysql.connection.cursor()
    myCursor.execute("select*from crops")
    data = myCursor.fetchall()
    myCursor.close()
    # return data will show an error do not try this 
    return render_template('startingtemplates/index.html', data = data)
@app.route("/about")
def about():
    return render_template("startingtemplates/about.html")
@app.route('/crops')
def crops():
    myCursor = mysql.connection.cursor()
    myCursor.execute("select*from crops")
    data = myCursor.fetchall()
    myCursor.close()
    # return data will show an error do not try this 
    return render_template('startingtemplates/crops.html', data = data)

@app.route("/team")
def team():
    return render_template("startingtemplates/team.html")
@app.route("/clients")
def clients():
    return render_template("startingtemplates/clients.html")
@app.route("/testing")
def testing():
    return render_template("startingtemplates/testing.html")

@app.route("/contact")
def contact():
    return render_template("startingtemplates/contact.html")




# PREDICTION ROUTES/////////////////////////////////
# predict and /predict1 (2 wata routes)
@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['n'])
    P = int(request.form['p'])
    K = int(request.form['k'])
    temp = float(request.form['t'])
    humidity = float(request.form['h'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['r'])


    # numpy array ma convert gareko
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    predictionRFC = loaded_modelRFC.predict(single_pred)
    predictionNB = loaded_modelNB.predict(single_pred)
    predictionDT = loaded_modelDT.predict(single_pred)
    predictionKNN = loaded_modelKNN.predict(single_pred)

    # print(accuracy_valueRFC)
    # print(accuracy_valueNB)
    # print(accuracy_valueDT)
    # print(accuracy_valueKNN)

    crop_dict = {
    1: "Rice",
    2: 'Maize',
    3: 'Jute',
    4: 'Cotton',
    5: 'Coconut',
    6: 'Papaya',
    7: 'Orange',
    8: 'Apple',
    9: 'Muskmelon',
    10: 'Watermelon',
    11: 'Grapes',
    12: 'Mango',
    13: 'Banana',
    14: 'Pomegranate',
    15: 'Lentil',
    16: 'Blackgram',
    17: 'Mungbean',
    18: 'Mothbeans',
    19: 'Pigeonpeas',
    20: 'Kidneybeans',
    21: 'Chickpea',
    22: 'Coffee'
    }
    if predictionRFC[0] in crop_dict:
        crop = crop_dict[predictionRFC[0]]
        resultRFC = "{} is the best crop".format(crop)
    else:
        resultRFC = "Please try again..."

    if predictionNB[0] in crop_dict:
        # print("name"+ str(predictionNB[0]))
        # print(predictionNB)
        crop = crop_dict[predictionNB[0]]
        resultNB =  "{} is the best crop".format(crop)
    else:
        resultNB = "Please try again..."

    if predictionDT[0] in crop_dict:
        crop = crop_dict[predictionDT[0]]
        resultDT =  "{} is the best crop".format(crop)
    else:
        resultDT = "Please try again..."


    if predictionKNN[0] in crop_dict:
        crop = crop_dict[predictionKNN[0]]
        resultKNN =  "{} is the best crop".format(crop)
    else:
        resultKNN = "Please try again..."


    # return render_template('index.html', resultNB = resultNB)

    return render_template(
        'admintemplates/testing.html',
        resultRFC = resultRFC,
        resultNB = resultNB,
        resultDT = resultDT,
        resultKNN = resultKNN,
        accuracy_valueNB = accuracy_valueNB*100,
        accuracy_valueKNN = accuracy_valueKNN*100,
        accuracy_valueDT = accuracy_valueDT*100,
        accuracy_valueRFC = accuracy_valueRFC*100,
    )

@app.route('/predict1', methods=['POST'])
def predict1():
    N = int(request.form['n'])
    P = int(request.form['p'])
    K = int(request.form['k'])
    temp = float(request.form['t'])
    humidity = float(request.form['h'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['r'])
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    predictionRFC = loaded_modelRFC.predict(single_pred)
    predictionNB = loaded_modelNB.predict(single_pred)
    predictionDT = loaded_modelDT.predict(single_pred)
    predictionKNN = loaded_modelKNN.predict(single_pred)
    crop_dict = {
    1: "Rice",
    2: 'Maize',
    3: 'Jute',
    4: 'Cotton',
    5: 'Coconut',
    6: 'Papaya',
    7: 'Orange',
    8: 'Apple',
    9: 'Muskmelon',
    10: 'Watermelon',
    11: 'Grapes',
    12: 'Mango',
    13: 'Banana',
    14: 'Pomegranate',
    15: 'Lentil',
    16: 'Blackgram',
    17: 'Mungbean',
    18: 'Mothbeans',
    19: 'Pigeonpeas',
    20: 'Kidneybeans',
    21: 'Chickpea',
    22: 'Coffee'
    }
    if predictionRFC[0] in crop_dict:
        crop = crop_dict[predictionRFC[0]]
        resultRFC = "{} is the best crop".format(crop)
    else:
        resultRFC = "Please try again..."
    if predictionNB[0] in crop_dict:
        crop = crop_dict[predictionNB[0]]
        resultNB =  "{} is the best crop".format(crop)
    else:
        resultNB = "Please try again..."

    if predictionDT[0] in crop_dict:
        crop = crop_dict[predictionDT[0]]
        resultDT =  "{} is the best crop".format(crop)
    else:
        resultDT = "Please try again..."
    if predictionKNN[0] in crop_dict:
        crop = crop_dict[predictionKNN[0]]
        resultKNN =  "{} is the best crop".format(crop)
    else:
        resultKNN = "Please try again..."
    return render_template(
        'startingtemplates/testing.html',
        resultRFC = resultRFC,
        resultNB = resultNB,
        resultDT = resultDT,
        resultKNN = resultKNN,
        accuracy_valueNB = accuracy_valueNB*100,
        accuracy_valueKNN = accuracy_valueKNN*100,
        accuracy_valueDT = accuracy_valueDT*100,
        accuracy_valueRFC = accuracy_valueRFC*100,
    )



# ADMIN ROUTES/////////////////////////////////

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        return render_template('admintemplates/dashboard.html', sessioninfo = session['email'])
    else:
        return render_template('authtemplates/login.html', message = loginWarning)


@app.route("/admin/parameters")
def admin_parameters():
    if 'email' in session:
        return render_template("admintemplates/parameters.html")
    else:
        return render_template('authtemplates/login.html', message = loginWarning)

@app.route("/admin/testinomials")
def admin_testinomials():
    if 'email' in session:
        return render_template("admintemplates/testinomials.html")
    else:
        return render_template('authtemplates/login.html', message = loginWarning)
    


@app.route('/admin/feedback')
def admin_feedback():
    myCursor = mysql.connection.cursor()
    myCursor.execute("select*from feedbacks")
    data = myCursor.fetchall()
    myCursor.close()
    if 'email' in session:
        return render_template('/admintemplates/feedback.html', data = data)
    else:
        return render_template('authtemplates/login.html', message = loginWarning)


@app.route('/admin/crops')
def admin_crops():
    myCursor = mysql.connection.cursor()
    myCursor.execute("select*from crops")
    data = myCursor.fetchall()
    myCursor.close()
    # return data will show an error do not try this 
    if 'email' in session:
        return render_template('/admintemplates/crops.html', data = data)
    else:
        return render_template('authtemplates/login.html', message = loginWarning)



@app.route("/admin/addcrops")
def addcrops():
    return render_template("entryforms/addcrops.html")


@app.route("/admin/testing")
def admin_testing():
    if 'email' in session:
        return render_template("admintemplates/testing.html")
    else:
        return render_template('admintemplates/login.html', message = loginWarning)

@app.route('/addcrops', methods = ["POST", "GET"])
def create():
    if request.method == "POST":
        name = request.form['name']
        scientific_name = request.form['scientific_name']
        description = request.form['description']
        image_url = request.form['image_url']
        season = request.form['season']
        myCursor = mysql.connection.cursor()
        myCursor.execute("INSERT INTO crops(name, scientific_name, description, season, image_url) values(%s, %s, %s, %s, %s)",(name, scientific_name, description, season, image_url))
        mysql.connection.commit()
        myCursor.close()
        return redirect(url_for('admin_crops'))




# CHATTING ROUTES///////////////////////////

@app.route("/get", methods = ["GET", "POST"])
def chat():
    input1 = request.form["msg"]
    print("Actual INPUT")
    print(input1)
    print("Lower INPUT")
    input2 = input1.lower()
    print(input2)
    print(" Text without PUNCUATION")
    input3 = input2.translate(str.maketrans('', '', string.punctuation))
    print(input3)
    print("Text without space")
    input4 = input3.replace(" ", "")
    print(input4)
    return get_Chat_response(input4)

def get_Chat_response(text):
    # IndentationError: expected an indented block after 'for' statement on line 468
    for step in range(5):
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        # return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
         
        if(text=="hello" or text=="hy" or text=="hey"):
            varResponse = "Hello Please, How may I help you !"
        elif(text=="accuracy" or text=="whatisaccuracy"):
            varResponse = "In a crop recommendation system, accuracy refers to the measure of how accurately the system predicts or recommends the appropriate crops for a given set of conditions or parameters. It is a performance metric that evaluates the correctness of the system's recommendations."
        elif(text=="cms" or text=="whatiscroprecommendationsystem" or text=="croprecommendationsystem" or text=="croprecommendation"):
            varResponse = "Answer of Crop recommendation system"
        elif(text=="whatiscrop" or text=="crop"):
            varResponse = "A crop recommendation system is an application of data-driven techniques and algorithms that help farmers to make informed decisions about which crops to plant or cultivate in a given area."
        elif(text=="rice" or text=="what is rice"):
            varResponse = "Rice is a staple food for most of the world's population. Whole grain, rice has more nutrients and health benefits than white rice."
        else:
            varResponse = "Sorry, I cant understand,\n Please ask project related queries.\nThank you"
        return varResponse
    




if __name__ == '__main__':
    app.run(debug=True, port="8000")
