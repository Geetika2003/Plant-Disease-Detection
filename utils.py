# Importing Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image


# Cleaning image    
def clean_image(image):
    image = np.array(image)
    
    # Resizing the image using the LANCZOS resampling filter
    image = np.array(Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS))
        
    # Adding batch dimensions to the image
    # You are setting :3, that's because sometimes the user uploads a 4-channel image,
    # So we just take the first 3 channels
    image = image[np.newaxis, :, :, :3]
    
    # Ensuring the image is of type float32
    image = image.astype(np.float32)
    
    return image
    

def get_prediction(model, image):
    # Creating a generator that yields batches of images
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # The flow method is designed to output batches of images
    test = datagen.flow(image)
    
    # Since `test` yields a batch, we get the first batch using `next()`
    batch = next(test)
    
    # Making the prediction
    predictions = model.predict(batch)
    predictions_arr = np.argmax(predictions, axis=1)
    
    return predictions, predictions_arr
     

# Making the final results 
def make_results(predictions, predictions_arr):
    result = {}
    if int(predictions_arr[0]) == 0:
        result = {"status": " is Healthy ",
                  "prediction": f"{int(predictions[0][0].round(2)*100)}%"}
    if int(predictions_arr[0]) == 1:
        result = {"status": ' has Multiple Diseases ',
                  "prediction": f"{int(predictions[0][1].round(2)*100)}%"}
    if int(predictions_arr[0]) == 2:
        result = {"status": ' has Rust ',
                  "prediction": f"{int(predictions[0][2].round(2)*100)}%"}
    if int(predictions_arr[0]) == 3:
        result = {"status": ' has Scab ',
                  "prediction": f"{int(predictions[0][3].round(2)*100)}%"}
    
    return result
