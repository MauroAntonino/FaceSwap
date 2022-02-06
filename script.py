import requests 
from base64 import encodebytes, b64decode
from PIL import Image
import io
import matplotlib.pyplot as plt

addr = 'http://192.168.128.2:5000'
face_swap_url = addr + '/api/face_swap'


def encode_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

# send http request with image and receive response
response = requests.post(face_swap_url,  json={'face':encode_image('./source.jpeg'),'body':encode_image('./destination.jpeg')})

img = response.json()['image']
img = b64decode(img)  
img = Image.open(io.BytesIO(img))
plt.imshow(img)