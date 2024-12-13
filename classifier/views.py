from django.shortcuts import render
import os
import uuid
from django.conf import settings
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model only once
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(settings.BASE_DIR, 'classifier', 'brain_classifier_final.h5')
        model = tf.keras.models.load_model(model_path)
    return model

def classify_image(request):
    if request.method == "POST" and request.FILES.get('brain_image'):
        brain_image = request.FILES['brain_image']

        # Save the image with a unique name
        unique_filename = f"{uuid.uuid4()}_{brain_image.name}"
        image_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
        with open(image_path, 'wb') as f:
            for chunk in brain_image.chunks():
                f.write(chunk)

        # Preprocess the image
        try:
            img = Image.open(image_path).resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Make prediction
            model = get_model()
            prediction = model.predict(img_array)
            tumor_type = int(prediction[0][0] > 0.5)  # 0 for Brain, 1 for No Brain
            confidence = float(prediction[0][0])

        except Exception as e:
            return render(request, 'classifier/result.html', {'error': f"Error processing image: {e}"})

        image_url = settings.MEDIA_URL + unique_filename
        return render(request, 'classifier/result.html', {
            'tumor_type': tumor_type,
            'confidence': confidence,
            'image_url': image_url
        })
    return render(request, 'classifier/upload.html')