from django.shortcuts import render
import os
import uuid
from django.conf import settings
import tensorflow as tf
import numpy as np
from PIL import Image


def classify_brain_image(model, img_array):
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Brain", prediction[0][0]
    else:
        return "Not Brain", 1 - prediction[0][0]

def classify_tumor_image(model, img_array):
 
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Tumor", prediction[0][0]
    else:
        return "No Tumor", 1 - prediction[0][0]

# Load the model only once
brain_model = None
tumor_model = None

def get_brain_model():
    global brain_model
    if brain_model is None:
        model_path = os.path.join(settings.BASE_DIR, 'classifier', 'brain_classifier_final.h5')
        try:
            brain_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load brain model: {e}")

    return brain_model

def get_tumor_model():
    global tumor_model
    if tumor_model is None:
        model_path = os.path.join(settings.BASE_DIR, 'classifier', 'tumor_classifier_final.h5')
        try:
            tumor_model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tumor model: {e}")
    return tumor_model

def classify_image(request):
    if request.method == "POST" and request.FILES.get('brain_image'):
        brain_image = request.FILES['brain_image']

        if not brain_image.content_type.startswith('image/'):
            return render(request, 'classifier/upload.html', {'error': "Uploaded file is not a valid image."})

        # Save the image with a unique name
        unique_filename = f"{uuid.uuid4()}_{brain_image.name}"
        image_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
        with open(image_path, 'wb') as f:
            for chunk in brain_image.chunks():
                f.write(chunk)

        # معالجة الصورة
        try:
            img = Image.open(image_path).resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # المرحلة الأولى: تصنيف الصورة
            brain_model = get_brain_model()
            brain_result, brain_prob = classify_brain_image(brain_model, img_array)

            tumor_result = "N/A"
            tumor_prob = 0.0

            if brain_result == "Brain":
                # المرحلة الثانية: تصنيف الورم
                tumor_result, tumor_prob = classify_tumor_image(tumor_model, img_array)
            
            image_url = settings.MEDIA_URL + unique_filename

            context_view = {
                'brain_result':brain_result,
                'confidence_brain':f"{brain_prob * 100:.2f}%",
                'tumor_result': tumor_result,
                'confidence_tumor': f"{tumor_prob * 100:.2f}%",
                'image_url': image_url
            }

            return render(request, 'classifier/result.html',context = context_view)
        
        except Exception as e:
            return render(request, 'classifier/result.html', {'error': f"Error processing image: {e}"})      
          
    return render(request, 'classifier/upload.html')