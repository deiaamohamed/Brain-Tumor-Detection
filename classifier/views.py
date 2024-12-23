from django.shortcuts import render
import os
import io
import base64
import logging
import uuid
from django.conf import settings
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from classifier.models import CNNmodel  

logger = logging.getLogger(__name__)

model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(settings.BASE_DIR, 'classifier', 'cnn_model.pth')  
        model = CNNmodel() 
        model.load_state_dict(torch.load(model_path))  # Load the trained model weights (the pth extensino as we used pytorch)
        model.eval()  
    return model

def classify_image(request):
    if request.method == "POST" and request.FILES.get('brain_image'):
        brain_image = request.FILES['brain_image']
        unique_filename = f"{uuid.uuid4()}_{brain_image.name}"
        image_path = os.path.join(settings.MEDIA_ROOT, unique_filename)
        with open(image_path, 'wb') as f:
            for chunk in brain_image.chunks():
                f.write(chunk)
        # Preprocess 
        try:
            # Resize the image to 128x128, i hate my life 
            img = Image.open(image_path).convert('RGB').resize((128, 128))
            
            # Apply transformations: عملت كدة رغم اني ظابطه في كود الترين للموديل بس عشان اتأكد
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ])
            img_tensor = transform(img).unsqueeze(0) 

            # Make prediction
            model = get_model()
            with torch.no_grad():
                prediction = model(img_tensor)
                prediction = torch.sigmoid(prediction)  # 0 tumor 1 no tumor i handled it in the result 
                tumor_type = torch.argmax(prediction, dim=1).item()  # Get the class with the nearset value (0,1)
                confidence = prediction[0, tumor_type].item()  # Get the confidence (probability) for the predicted class

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render(request, 'classifier/result.html', {'error': "Failed to classify the image."})

        image_url = settings.MEDIA_URL + unique_filename
        contextPage={
            'tumor_type': tumor_type,
            'confidence': confidence,
            'image_url': image_url
        }
        return render(request, 'classifier/result.html', contextPage)
    return render(request, 'classifier/upload.html')

# =================================================================================================

def Visualization(request):

    
    contextPage = {
        'View a network with multiple samples of each (Healthy).':'media\Healthy_Brain_Grid.png',
        'View a network with multiple samples of each (Tumor).':'media\Tumor_Brain_Grid.png',
        'Dimensional reduction of image representation in two-dimensional space':'media\PCA_Visualization.png',
        'Use Boxplot to compare dimensions.':'media\Boxplot_of_Image_Widths_by_Category.png',
        'Analysis of distribution of pixel values (intensity) for images (Healthy)':'media\Pixel_ntensity_Distribution_(Healthy).png',
        'Analysis of distribution of pixel values (intensity) for images (Tumor)':'media\Pixel_Intensity_Distribution_(Tumor).png',  
        'Analyze the number of images in each category (Healthy/tumor).':r'media\Number_of Images_in_Each_Category.png',
        'Analyze the dimensions of images to see the diversity in their size':'media\Distribution_of_Image_Widths.png',
        'Confusion Matrix of the Model':r"media\confusion.png"
    }
    # print(contextPage)


    return render(request, 'classifier/visualize.html',{'contextPage': contextPage})
