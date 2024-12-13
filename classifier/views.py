from django.shortcuts import render
import os
from django.conf import settings
#import tensorflow as tf
import numpy as np
from PIL import Image


# model = tf.keras.models.load_model('classifier/ai_models/brain_classifier.h5')

def classify_image(request):
    if request.method == "POST" and request.FILES.get('brain_image'):
        brain_image = request.FILES['brain_image']

        # حفظ الصورة في مجلد media
        image_path = os.path.join(settings.MEDIA_ROOT, brain_image.name)
        with open(image_path, 'wb') as f:
            for chunk in brain_image.chunks():
                f.write(chunk)
        
        # إرسال الصورة ورابطها للقالب
        image_url = settings.MEDIA_URL + brain_image.name

        # معالجة الصورة
        img = Image.open(image_path).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # توقع الموديل
        # prediction = model.predict(img_array)
        # tumor_type = np.argmax(prediction)

        return render(request, 'classifier/result.html',{'tumor_type': 'tumor_type', 'image_url': image_url})
    return render(request, 'classifier/upload.html')


