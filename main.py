import cv2 as cv
import cvzone
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# capture frames from the pc's webcam
capture = cv.VideoCapture(0)
# an object to capture only one face
face_mesh_detector = FaceMeshDetector(maxFaces=1)


# the main function, or entry point
def main():
    # get some text from the user and display it on the second screen
    """
    # still thinking of the best way to get the user's input, and add the words to the textList variable with the max number of words being 4
    # ex of input: "I am a good boy and i live in the ocean"
    # output: ["I am a good", "boy and i live", "in the ocean"]

    user_text = input("Enter the text to be shown: ")
    if user_text is not None:
        textList = []
        count = -1
        x= ""
        for idx, word in enumerate(user_text.split(' ')):
            # divide the user's text into series of four
            if count == 3:
                x += word
                textList.append(x)
                x = ""
                count=-1
            else:
                x += word
                count+=1

    else:
        textList = ["The font size of", "this text changes with", "respect to the distance", "between me and",
                    "the camera(depth)!"]
    """

    # If you change this text.. your words should follow a series a four or three as below. (For better display)
    textList = ["The font size of", "this text changes with", "respect to the distance(depth)", "between you the user",
                "and the camera!"]

    # enter the infinite loop
    while True:
        # success variable indicates whether our program succeeded in reading the webcam's next frame (True or False)
        success, image = capture.read()
        # an empty black board where the text resides
        white_board = np.zeros_like(image)

        if success:
            # get the faces in present
            image, faces = face_mesh_detector.findFaceMesh(image, draw=False)

            if faces:
                # get the first face present in the webcam's view-area
                face_found = faces[0]
                left_eye_pts, right_eye_pts = face_found[145], face_found[374]
                distance_btw_user_eyes, _ = face_mesh_detector.findDistance(left_eye_pts, right_eye_pts)

                W = 6.3  # supposed distance between two human eyes (by scientists of the eye)
                # d = 50      # assumed distance between the camera and the object f = (distance_btw_eyes * d) / W
                # focal length: distance between the image from the camera and the camera's lens
                f = 820
                # calculate the depth or the distance between the computer and the user
                depth = (f * W) / distance_btw_user_eyes

                # add a text indicating this Depth/distance
                cvzone.putTextRect(image, f"Distance/depth = {int(depth)}cm", (25, 50),
                                   thickness=2, scale=1.5)

                font_size = 0.3 + (int(depth / 10) * 10) / 85  # we adjust the fontsize based on the depth

                # loop through the contents of the list containing our full text
                for idx, t in enumerate(textList):
                    # set a base height for our text
                    base_height = 17 + int(depth / 5)
                    cv.putText(white_board, t, (30, 52 + (idx * base_height)),
                               fontFace=cv.FONT_HERSHEY_DUPLEX,
                               fontScale=font_size, thickness=2, color=(255, 255, 255))

            # stack the images on one row (horizontal stacking)
            all_images = cvzone.stackImages([image, white_board], 2, .75)
            # open a window with a specified name
            cv.imshow("Face Distance Analyst", all_images)

        # break if the button 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release the webcam
    capture.release()
    # destroy all opened windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
