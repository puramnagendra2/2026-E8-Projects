#Import necessary libraries
from flask import Flask, render_template, request, jsonify
 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

trained_model = load_model('pred_cotton.h5')
print('@@ Model loaded')

def predict_disease(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['Aphids', 'Army Worm', 'Bacterial Blight', 'Healthy Leaf', 'Powdery Mildew', 'Target Spot']  # Class labels

    if class_labels[class_index] == "Aphids":
        return "Aphids", 'aphids.html'
    elif class_labels[class_index] == 'Army Worm':
         return "Army_Worm", "army_worm.html"
    elif class_labels[class_index] == 'Bacterial Blight':
         return "Bacterial Blight", "bacterial_blight.html"
    elif class_labels[class_index] == 'Healthy Leaf':
         return "Healthy Leaf", "healthy_leaf.html"
    elif class_labels[class_index] == 'Powdery Mildew':
         return "Powdery Mildew", "powdery_mildew.html"
    else:
         return "Target Spot", "target_spot.html"
     

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image']
        print("File is ", file)
        print("Allowed file ", allowed_file(file.filename))
        if file and allowed_file(file.filename):
            # Save file temporarily
            filepath = os.path.join("static/uploads/", file.filename)
            print("@@ Input posted = ", filepath)
            file.save(filepath)
 
            print("@@ Predicting class......")
            pred, output_page = predict_disease(trained_model, filepath)
               
            return render_template(output_page, pred_output = pred, user_image = filepath)
     
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False) 

