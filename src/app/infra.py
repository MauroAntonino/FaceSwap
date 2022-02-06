import dlib
import numpy as np
from matplotlib import pyplot as plt
from base64 import encodebytes
import io
from PIL import Image
import cv2
# https://livecodestream.dev/post/build-a-face-swapping-app-part-2-api/


class FaceSwap:
    def __init__(self, face, body, predictor) -> None:
        self.face = face
        self.body = body
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor)
    
    def main(self):
        face_gray, body_gray = self.get_images_gray()
        landmarks_points = self.get_landmarks(face_gray)
        points = self.get_points(landmarks_points)
        triangles = self.get_triangles(landmarks_points)
        face_cp = self.face.copy()
        
        indexes_triangles = self.get_index_triangles_face(triangles, face_cp, points)

        height, width, channels = self.body.shape
        landmarks_points2 = self.get_landmarks(body_gray)
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)
        body_new_face = np.zeros((height, width, channels), np.uint8)
        height, width = face_gray.shape
        lines_space_mask = np.zeros((height, width), np.uint8)

        body_new_face = self.big_func(indexes_triangles, landmarks_points, lines_space_mask, landmarks_points2, body_new_face)
        return self.func_2(body_gray, convexhull2, body_new_face)
        
    def get_points(self, landmarks_points):
        return np.array(landmarks_points, np.int32)

    def get_triangles(self, landmarks_points):
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points) 
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect) # Creates an instance of Subdiv2D
        subdiv.insert(landmarks_points) # Insert points into subdiv
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        return triangles
    
    def get_index(self, nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index
    
    def get_index_triangles_face(self, triangles, face_cp, points):
        indexes_triangles = []
        for triangle in triangles :

            # Gets the vertex of the triangle
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            
            # Draws a line for each side of the triangle
            cv2.line(face_cp, pt1, pt2, (255, 255, 255), 3,  0)
            cv2.line(face_cp, pt2, pt3, (255, 255, 255), 3,  0)
            cv2.line(face_cp, pt3, pt1, (255, 255, 255), 3,  0)

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = self.get_index(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = self.get_index(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = self.get_index(index_pt3)

            # Saves coordinates if the triangle exists and has 3 vertices
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                vertices = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(vertices)
        return indexes_triangles

    def get_image_mask(self, height, width):
        return np.zeros((height, width), np.uint8)

    def get_images_gray(self):
        face_gray = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
        body_gray = cv2.cvtColor(self.body, cv2.COLOR_BGR2GRAY)
        return face_gray, body_gray
    
    def get_landmarks(self, image_gray):
        rect = self.detector(image_gray)[0]
        landmarks = self.predictor(image_gray, rect)

        landmarks_points = [] 
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        return landmarks_points
    
    def big_func(self, indexes_triangles, landmarks_points, lines_space_mask, landmarks_points2, body_new_face):
        for triangle in indexes_triangles:

            # Coordinates of the first person's delaunay triangles
            pt1 = landmarks_points[triangle[0]]
            pt2 = landmarks_points[triangle[1]]
            pt3 = landmarks_points[triangle[2]]

            # Gets the delaunay triangles
            (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
            cropped_triangle = self.face[y: y+height, x: x+widht]
            cropped_mask = np.zeros((height, widht), np.uint8)

            # Fills triangle to generate the mask
            points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
            cv2.fillConvexPoly(cropped_mask, points, 255)

            # Draws lines for the triangles
            cv2.line(lines_space_mask, pt1, pt2, 255)
            cv2.line(lines_space_mask, pt2, pt3, 255)
            cv2.line(lines_space_mask, pt1, pt3, 255)

            lines_space = cv2.bitwise_and(self.face, self.face, mask=lines_space_mask)

            # Calculates the delaunay triangles of the second person's face

            # Coordinates of the first person's delaunay triangles
            pt1 = landmarks_points2[triangle[0]]
            pt2 = landmarks_points2[triangle[1]]
            pt3 = landmarks_points2[triangle[2]]

            # Gets the delaunay triangles
            (x, y, widht, height) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
            cropped_mask2 = np.zeros((height,widht), np.uint8)

            # Fills triangle to generate the mask
            points2 = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
            cv2.fillConvexPoly(cropped_mask2, points2, 255)

            # Deforms the triangles to fit the subject's face : https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
            points =  np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)  # Warps the content of the first triangle to fit in the second one
            dist_triangle = cv2.warpAffine(cropped_triangle, M, (widht, height))
            dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)

            # Joins all the distorted triangles to make the face mask to fit in the second person's features
            body_new_face_rect_area = body_new_face[y: y+height, x: x+widht]
            body_new_face_rect_area_gray = cv2.cvtColor(body_new_face_rect_area, cv2.COLOR_BGR2GRAY)

            # Creates a mask
            masked_triangle = cv2.threshold(body_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])

            # Adds the piece to the face mask
            body_new_face_rect_area = cv2.add(body_new_face_rect_area, dist_triangle)
            body_new_face[y: y+height, x: x+widht] = body_new_face_rect_area
        return body_new_face
        
    def func_2(self, body_gray, convexhull2, body_new_face):

        body_face_mask = np.zeros_like(body_gray)
        body_head_mask = cv2.fillConvexPoly(body_face_mask, convexhull2, 255)
        body_face_mask = cv2.bitwise_not(body_head_mask)

        body_maskless = cv2.bitwise_and(self.body, self.body, mask=body_face_mask)
        result = cv2.add(body_maskless, body_new_face)

        # Gets the center of the face for the body
        (x, y, widht, height) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x+x+widht)/2), int((y+y+height)/2))
        seamlessclone = cv2.seamlessClone(result, self.body, body_head_mask, center_face2, cv2.NORMAL_CLONE)
        return self.sendResponse(seamlessclone)
        # cv2.imwrite("./result.png", seamlessclone)
    
    def sendResponse(self, new):
        new = Image.fromarray(new)
        byte_arr = io.BytesIO()
        new.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        
        return {'image': encoded_img}

# face = cv2.imread("walter.jpeg")
# body = cv2.imread("cr7.jpeg")
# obj = FaceSwap(face=face, body=body, predictor="./shape_predictor_68_face_landmarks.dat")
# obj.main()