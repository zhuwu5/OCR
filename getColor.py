import cv2
import numpy as np

def get_color(image):
    color_result = False
    # BGR
    # set red thresh
    # lower_blue = np.array([156, 43, 46])
    # upper_blue = np.array([180, 255, 255])

    # set black thresh
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([170, 255, 110])

    img = image

    # get a frame and show

    frame = img

    # cv2.imshow('Capture', frame)

    # change to hsv model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('hsv.jpg', hsv)

    # get mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('Mask', mask)
    if np.sum(mask) != 0:
        color_result = True

    # detect red
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('Result', res)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return color_result


def get_whether_blackdot(im):

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 250

    params.filterByColor = True
    params.blobColor = 0
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(im)
    print(len(keypoints))

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)

    cv2.waitKey(0)
    return 'True'



if __name__ == '__main__':
    img = cv2.imread(r'D:\ocrTestImage\cover1.png')
    get_whether_blackdot(img)
