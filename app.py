from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from keras.preprocessing.image import img_to_array

app = Flask(__name__)



model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    color_me = []

    i = image.load_img(img_path, target_size=(256, 256))
    color_me.append(img_to_array(i))
    color_me = np.array(color_me, dtype=float)
        
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

    output = model.predict(color_me)
    return output,color_me


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename    
        img.save(img_path)

        output,color_me = predict_label(img_path)
        output = output * 128

        # Output colorizations
        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:,:,0] = color_me[i][:,:,0]
            cur[:,:,1:] = output[i]
            img_path2 = "static/167.jpg"
            imsave(img_path2, lab2rgb(cur))
            
    return render_template("index.html", prediction = "Done", img_path = img_path,
                           img_path2 = img_path2)


if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)
