import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt


def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left()-60, x.top()-60,
                    x.right()+60, x.bottom()+60) for x in detected_faces]

    return face_frames

# Load image (Change input address here)
img_path = 'messi.jpg'
image = io.imread(img_path)

# Detect faces
detected_faces = detect_faces(image)

# Crop faces and plot
for n, face_rect in enumerate(detected_faces):
    if n > 1:
        break
    face = Image.fromarray(image).crop(face_rect)
    face = face.resize((96,96), Image.ANTIALIAS)
    plt.subplot(1, len(detected_faces), n+1)
    plt.axis('off')
    plt.imshow(face)
    plt.show()

    #Output (Change output image address here)
    face.save('res.jpg')


