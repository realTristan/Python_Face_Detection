import cv2, random

# // Set contant values
RGB_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]
IMAGE = cv2.imread('files/people.jpg')

# Return cascaded image function
def image_casc():
    return cv2.CascadeClassifier("files/haarcascade.xml").detectMultiScale(
        cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY), 1.1, 4
    )

# // Find faces function
def find_faces(faces:list, face_count:int):
    for (x, y, w, h) in faces:
        color = random.choice(RGB_COLORS)
        face_count+=1
        
        # // Draw a rectangle on the persons face
        cv2.putText(IMAGE, str(face_count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(IMAGE, (x, y), (x+w, y+h), color, 2)


# // Run the functions and show the image
if __name__ == "__main__":
    faces = image_casc()
    find_faces(faces, 0)
    
    cv2.imshow("Python Face Detection", IMAGE)
    cv2.waitKey()