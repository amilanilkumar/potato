import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image,ImageOps
import numpy as np
import os 

from io import BytesIO 
st.cache(allow_output_mutation=True)
CLASS_NAMES=["early Blight ","late Blight", "healthy"]
def load_model():
    model=tf.keras.models.load_model('./model/1')
    return model
model=load_model()

st.write("""#potato leaf disease classification""")
file=st.file_uploader("please upload a potato file",type=["jpg","png"])

def import_and_predict(image_data , model):
    size=(256,256)
    image= ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image=np.asarray(image)
    img_reshape = np.expand_dims(image,axis=0)
    prediction = model.predict(img_reshape,batch_size=1)
    return prediction


if file is None:
    st.text("please upload an image file")
else:
        image=Image.open(file)
        st.image(image, use_column_width=True)
        predictions= import_and_predict(image,model)
        index=np.argmax(predictions[0])
        predicted_class=CLASS_NAMES[index]
        confidenc=np.max(predictions[0])
        st.write(predicted_class)
        st.write(confidenc)
        print(
            "this image belomgs to{} with{:.2f} percent."
            .format(CLASS_NAMES[np.argmax(confidenc)],100* np.max(confidenc))
        )    