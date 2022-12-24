"""
    We want to find the homography between two images.
    
    We will use an ORB detector to find the keypoints and descriptors.
    we will then use matching keypoints to find the homography.
"""

import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, filename="out.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
logging.info("Starting...")

""" Histogram equalization of an image.

    Args:
        rgb_img: RGB image to be equalized.

    Returns:
        equalized_img: Histogram equalized image.

    In order to provide some degree of illumination invariance in our ORB-detector
    based alignment, we will perform histogram equalization on the Y channel of the
    images, which is the luminance channel.

    by having a more uniform illumination across the single image and similar 
    image distributions across the pairs, we can expect to have better results, 
    we will especially reduce drastically the effect of shadows and bright spots, 
    slashing down on the amount of false positives.

    """


def run_histogram_equalization(rgb_img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img


if __name__ == "__main__":

    filenames = ["A_1.png",
                 "A_2.png"]

    try:
        # create a 800x600 window

        aspect_ratio = 9/16

        width = 800
        heigth = int(width * aspect_ratio)

        img1 = cv2.imread(filenames[0])

        img2 = cv2.imread(filenames[1])

        img1_norm = run_histogram_equalization(img1)
        img2_norm = run_histogram_equalization(img2)

        logging.info("Images loaded, continuing.")

        logging.info(f"Image 1: {img1_norm.shape}, Image 2: {img2_norm.shape}")
        logging.info(f"Image 1: {img1_norm.dtype}, Image 2: {img2_norm.dtype}")

        # create window
        cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img1", width, heigth)

        cv2.imshow("img1", img1_norm)
        cv2.waitKey(0)

        cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("img2", width, heigth)

        cv2.imshow("img2", img2_norm)
        cv2.waitKey(0)

        #cv2.rotate(cv2.imread(filenames[1]), cv2.ROTATE_90_CLOCKWISE)
        #cv2.warpAffine(img1, np.float32([[1, 0, 0], [0, 1, 0]]), (img1.shape[1], img1.shape[0]))

        # Initiate ORB detector
        orb = cv2.ORB_create()

        # check if the images are the same size
        if img1_norm.shape != img2_norm.shape:
            logging.error("Images are not the same size, aborting.")
            raise Exception("Images are not the same size, aborting.")
        else:
            logging.info("Images are the same size, continuing.")
            logging.info(
                f"Image 1: {img1_norm.shape}, Image 2: {img2_norm.shape}")

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1_norm, None)
        kp2, des2 = orb.detectAndCompute(img2_norm, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Match descriptors.
        matches = bf.match(des1, des2)

        logging.info("Found %d matches", len(matches))

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 32 matches.

        MATCHES_TO_CONSIDER = 32
        matches = matches[:MATCHES_TO_CONSIDER]

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

        # create window
        cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Matches", width, heigth)

        cv2.imshow("Matches", img3)
        cv2.waitKey(0)

        # compute homography
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        dst = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

        # create window

        cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Warped", dst.shape[1]//8, dst.shape[0]//8)

        cv2.imshow("Warped", dst)

        cv2.waitKey(0)
    except:

        logging.exception("Error occurred, cleaning up...")
    finally:

        cv2.destroyAllWindows()
        logging.info("Done.")
