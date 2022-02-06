from flask import request, Response
from app import app
import base64
from PIL import Image
import numpy as np
import io
from app.infra import FaceSwap

@app.route('/api/face_swap', methods=['POST'])
def swap():
    r = request
    
    # decode image
    face = base64.b64decode(r.json.get('face'))   
    face = Image.open(io.BytesIO(face))

    body = base64.b64decode(r.json.get('body'))
    body = Image.open(io.BytesIO(body))

    # convert image to numpy array for processing
    face = np.array(face)
    body = np.array(body)
    obj = FaceSwap(face=face, body=body, predictor="app/shape_predictor_68_face_landmarks.dat")
    return obj.main()