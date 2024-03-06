import os
import requests
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO


def download_image(url, file_path):
    try:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        response = requests.get(url)
        if response.status_code == 200:
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1]
            if not file_extension:
                file_extension = '.jpg'
            file_path_with_extension = file_path + file_extension
            image = Image.open(BytesIO(response.content))
            image.save(file_path_with_extension)
            
            return {"succes": True}
        else:
            return {"succes": False}
    except Exception as e:
        print("An error occurred:", str(e))
        return str(e)
    
def get_ready_for_download(data):
    try:
        for detail in data['details']:
            name = detail['name']
            image_url = detail['image']
            className = data['className']
            file_path = f"D:/Professional/Project/FinalYr/attendenceSystem/automated_attendense_system/images/{className}/{name}.jpg"
            download_image(image_url, file_path)
        return {"succes": True}
    
    except Exception as e:
        print("An error occurred:", str(e))
        return {"succes": False}