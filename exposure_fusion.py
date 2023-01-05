
import datetime
import numpy as np
import cv2
import logging
import hashlib


from builtins import isinstance

#import dataclass
from dataclasses import dataclass


@dataclass
class Exponents():
    """Exponents structure wrapping the three RGB exponents for the HDR Fusion algorithm.
    """
    e_contrast: float
    e_saturation: float
    e_exposedness: float


class ExposureFusion():
    """ExposureFusion Functor for the HDR Fusion algorithm.

    """

    def __init__(self, perform_alignment: bool = True,
                 exponents: Exponents = Exponents(1., 1., 1.), sigma: float = 0.2,
                 matches_to_consider: int = 32, pyramid_levels: int = 3):
        """__init__ Sets the parameters for the ExposureFusion functor.

        Parameters
        ----------
        perform_alignment : bool, optional
            Whether to perform image alignment between the input LDR images as a 
            preprocessing step, by default True
        exponents : Exponents, optional
            The three exponents for the (R,G,B) channels, by default Exponents(1., 1., 1.),
            as suggested in the paper
        sigma : float, optional
            The standard deviation of the Gaussian used to calculate
            well-exposedness, gets squared to variance internally, by default 0.2
        matches_to_consider : int, optional
            The number of matches to consider when performing image alignment, 
            by default 32
        pyramid_levels : int, optional
            The number of levels to use in the Gaussian and Laplacian pyramids, by default 3,
            deeper pyramids introduce more artifacts.
        """

        self.perform_alignment: bool = perform_alignment
        self.exponents: Exponents = exponents
        self.sigma: float = sigma ** 2
        self.matches_to_consider: int = 32
        self.pyramid_levels: int = pyramid_levels

        """
            Initialize the logger for the ExposureFusion functor.
        """

        self.logger = logging.getLogger(hashlib.md5(
            str(datetime.datetime.now()).encode()).hexdigest())

        self.logger.handlers = []

        self.time_creation = str(datetime.datetime.now()).replace(
            ":", "-").replace(" ", "_").replace(".", "_").replace("-", "_").replace("/", "_")
        fileHandler = logging.FileHandler(
            f"logs/out_{self.time_creation}.log", mode="w")
        streamHandler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        datefmt = "%d-%b-%y %H:%M:%S"

        formatter.datefmt = datefmt

        # set formatter and datefmt
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self.logger.addHandler(streamHandler)
        self.logger.addHandler(fileHandler)
        self.logger.setLevel(logging.INFO)

        assert isinstance(
            self.exponents, Exponents), "exponents must be of type Exponents"
        assert self.sigma > 0, "sigma must be positive"
        assert self.matches_to_consider > 0, "matches_to_consider must be positive"
        assert self.pyramid_levels >= 1, "pyramid_levels must be at least 1"

        if self.perform_alignment:
            import cv2
            """
                ORB is open source, fast and generally performs better than SIFT on
                this task.
            """
            self.ORB_detector = cv2.ORB_create()

            # create a Brute Force Matcher object, using Hamming distance since
            # We are using ORB.
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __repr__(self) -> str:
        """__repr__ Returns a string representation of the ExposureFusion functor.

        Returns
        -------
        str
            A string representation of the ExposureFusion functor.
        """
        return f"""ExposureFusion(perform_alignment={self.perform_alignment}, exponents={self.exponents}, sigma={self.sigma}, matches_to_consider={self.matches_to_consider})"""

    def __call__(self, images: "list[np.ndarray]") -> np.ndarray:
        """__call__ Perform exposure fusion

        Parameters
        ----------
        images : list[np.ndarray]
            A list of numpy arrays, all of the same shape and with RGB channels last.
            the individual images should be of the same scene at different levels of exposure.

            The images should be at least 2.

        Returns
        -------
        np.ndarray
            The resulting HDR image obtained by applying Exposure Fusion on the LDR inputs
        """

        # Safety checks
        try:
            assert len(images) >= 2
            assert all([isinstance(image, np.ndarray) for image in images])
            assert all([image.shape == images[0].shape for image in images])
            assert all([image.shape[-1] == 3 for image in images])
        except AssertionError as e:
            self.logger.exception("Invalid input to ExposureFusion functor")
            return None

        self.logger.info(
            "Input images are valid, proceeding with Exposure Fusion")

        self.logger.info(
            f"""Processing {len(images)} images, with shape {images[0].shape}""")

        # Deep copy the images to avoid modifying the originals
        images = [image.copy() for image in images]

        if self.perform_alignment:
            self.logger.info("Performing image alignment")
            try:
                images = self.align_images(images)
            except Exception as e:
                self.logger.exception("Image alignment failed")
                return None
        else:
            self.logger.info("Skipping image alignment")

        self.logger.info("Calculating weights")

        try:
            weights = self.calculate_weights(images)
        except Exception as e:
            self.logger.exception("Failed to calculate weights")
            return None

        self.logger.info("Creating image pyramids")

        try:
            gaussians, laplacians = self.create_image_pyramids(images, weights)
        except Exception as e:
            self.logger.exception("Failed to create image pyramids")
            return None

        self.logger.info("Blending pyramids into final Laplacian")

        try:
            final_laplacian = self.blend_pyramids(gaussians, laplacians)
        except Exception as e:
            self.logger.exception("Failed to blend pyramids")
            return None

        self.logger.info("Reconstructing final HDR image")

        try:
            hdr_image = self.reconstruct_image(final_laplacian)
        except Exception as e:
            self.logger.exception("Failed to reconstruct HDR image")
            return None

        return hdr_image

    def align_images(self, in_images: "list[np.ndarray]") -> "list[np.ndarray]":
        """align_images Performs image alignment on the input images

        This method is not meant to be called directly, but rather as a preprocessing step
        in the functor's __call__ method pipeline.

        Parameters
        ----------
        in_images : list[np.ndarray]
            A list of numpy arrays, all of the same shape and with RGB channels last.
            the individual images should be of the same scene at different levels of exposure.

            The images should be at least 2.

        Returns
        -------
        list[np.ndarray]
            The aligned images
        """

        center_indx = len(in_images) // 2

        images = in_images.copy()

        """ 
            We perform the following preprocessing steps on the images:
            1. Histogram equalization
            2. Gaussian blur
            3. Grayscaling
            
            The histogram is equalized to improve the contrast of the image, the gaussian 
            blur is used to reduce noise and the grayscaling is used to reduce the number
            of features to match.
        """

        # Histogram equalization
        for idx, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            images[idx] = img

        # Gaussian blur
        for idx, img in enumerate(images):
            img = cv2.GaussianBlur(img, (3, 3), 0)
            images[idx] = img

        # Grayscaling
        for idx, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[idx] = img

        # Find the center image, which will be used to align the other images to
        reference_img = images[center_indx]

        kp_ref, des_ref = self.ORB_detector.detectAndCompute(
            reference_img, None)

        for idx, img in enumerate(images):
            if idx == center_indx:
                continue

            kp, des = self.ORB_detector.detectAndCompute(img, None)
            matches = self.matcher.match(des, des_ref)
            matches = sorted(matches, key=lambda x: x.distance)

            # only consider the K best matches, to reduce the effect of outliers.
            matches = matches[:self.matches_to_consider]

            src_pts = np.float32(
                [kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            img = cv2.warpPerspective(in_images[idx], M, img.shape[:2][::-1])

            in_images[idx] = img

        return in_images

    def calculate_weights(self, images: "list[np.ndarray]") -> "list[np.ndarray]":
        """calculate_weights Calculates the weights for each image in the input list

        This method is not meant to be called directly, but rather as a preprocessing step
        in the functor's __call__ method pipeline.

        Parameters
        ----------
        images : list[np.ndarray]
            A list of numpy arrays, all of the same shape and with RGB channels last.
            the individual images should be of the same scene at different levels of exposure.

            The images should be at least 2.

        Returns
        -------
        list[np.ndarray]
            The weights for each image in the input list
        """

        weights = []
        weights_sum = np.zeros(images[0].shape[:2], dtype=np.float32)

        # Calculate contrast, saturation and exposure weights

        for image in images:

            image = image.astype(np.float32) / 255.0

            w_c = self.calculate_contrast_weight(image)
            w_s = self.calculate_saturation_weight(image)
            w_e = self.calculate_exposure_weight(image)

            w = ((w_c ** self.exponents.e_contrast) + 1) * \
                ((w_s ** self.exponents.e_contrast) + 1) * \
                ((w_e ** self.exponents.e_exposedness) + 1)

            weights.append(w)

            weights_sum += w

        # Normalize weights
        weights = [np.uint8(255 * w / weights_sum)
                   for w in weights]

        return weights

    def calculate_contrast_weight(self, image: np.ndarray) -> np.ndarray:
        """calculate_contrast_weight Calculates the contrast weight for the input image

        Parameters
        ----------
        image : np.ndarray
            The image for which the contrast weight is to be calculated

        Returns
        -------
        np.ndarray
            The contrast weight for the input image
        """

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)

        # Calculate contrast
        contrast = np.abs(laplacian)

        return contrast

    def calculate_saturation_weight(self, image: np.ndarray) -> np.ndarray:
        """calculate_saturation_weight Calculates the saturation weight for the input image

        Parameters
        ----------
        image : np.ndarray
            The image for which the saturation weight is to be calculated

        Returns
        -------
        np.ndarray
            The saturation weight for the input image
        """

        # Calculate saturation
        saturation = image.std(axis=2)

        return saturation

    def calculate_exposure_weight(self, image: np.ndarray) -> np.ndarray:
        """calculate_exposure_weight Calculates the exposure weight for the input image

        Parameters
        ----------
        image : np.ndarray
            The image for which the exposure weight is to be calculated

        Returns
        -------
        np.ndarray
            The exposure weight for the input image
        """

        # Calculate exposure
        exposure = np.prod(
            np.exp(-((image - 0.5)**2)/(2*self.sigma)), axis=2, dtype=np.float32)

        return exposure

    def create_image_pyramids(self, images: "list[np.ndarray]",
                              weights: "list[np.ndarray]") -> "list[list[np.ndarray]]":
        """create_image_pyramids Generate gaussian and laplacian pyramids for the input images

        Parameters
        ----------
        images : list[np.ndarray]
            A list of input LDR images
        weights : list[np.ndarray]
            A list of weights for each image in the input list

        Returns
        -------
        (list[np.ndarray], list[np.ndarray])
            A list of gaussian and laplacian pyramids for the input images
        """

        g_pyramids = []
        l_pyramids = []

        for image, weight in zip(images, weights):
            gaussian_pyramid = []

            image_gaussians = []
            laplacian_pyramid = []

            # Create gaussian pyramid
            for i in range(self.pyramid_levels):
                if i == 0:
                    gaussian_pyramid.append(weight)
                else:
                    gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))

            # Display gaussian pyramid

            # Create gaussian pyramid for image
            for i in range(self.pyramid_levels):
                if i == 0:
                    image_gaussians.append(image)
                else:
                    image_gaussians.append(cv2.pyrDown(image_gaussians[-1]))

            # Create laplacian pyramid
            for i in range(self.pyramid_levels - 1, -1, -1):

                if i == self.pyramid_levels - 1:
                    laplacian_pyramid.append(image_gaussians[i])
                else:
                    size = (image_gaussians[i].shape[1],
                            image_gaussians[i].shape[0])
                    gaussian_expanded = cv2.pyrUp(
                        image_gaussians[i+1], dstsize=size)
                    laplacian_pyramid.append(cv2.subtract(
                        image_gaussians[i], gaussian_expanded))

            g_pyramids.append(gaussian_pyramid)
            l_pyramids.append(laplacian_pyramid)

        return g_pyramids, l_pyramids

    def blend_pyramids(self, gaussian_pyramids: "list[list[np.ndarray]]",
                       laplacian_pyramids: "list[list[np.ndarray]]") -> any:
        """blend_pyramids Blends the Gaussian and laplacian pyramids

        Parameters
        ----------
        gaussian_pyramids : list[np.ndarray]
            A list of gaussian pyramids, each of which has dimensions (pyramid_levels, height, width, channels)
        laplacian_pyramids : list[np.ndarray]
            A list of laplacian pyramids, each of which has dimensions (pyramid_levels, height, width, channels)

        Returns
        -------
        np.ndarray
            _description_
        """
        res_laplacian = []

        for level in range(self.pyramid_levels):

            reverse_level = self.pyramid_levels - (1 + level)

            res_plevel = np.zeros(laplacian_pyramids[0][reverse_level].shape,
                                  dtype=np.uint8)

            for img_idx in range(len(gaussian_pyramids)):

                gaussian = gaussian_pyramids[img_idx][level]
                laplacian = laplacian_pyramids[img_idx][reverse_level]

                gaussian = np.float32(gaussian/255)

                gaussian = np.repeat(gaussian[:, :, np.newaxis], 3, axis=2)
                combination = cv2.multiply(
                    gaussian, laplacian, dtype=cv2.CV_8UC3)
                res_plevel = cv2.add(res_plevel, combination)

            res_laplacian.append(res_plevel)

        return res_laplacian

    def reconstruct_image(self, laplacian_pyramid: "list[np.ndarray]") -> np.ndarray:
        """reconstruct_image Retrieves the final HDR image from the laplacian pyramid
        that was generated by the blend_pyramids function

        Parameters
        ----------
        laplacian_pyramid : list[np.ndarray]
            A list of images, each of which has dimensions (height, width, channels)

        Returns
        -------
        np.ndarray
            The final HDR image
        """

        laplacian_pyramid = laplacian_pyramid[::-1]

        res = laplacian_pyramid[0]

        for i in range(1, len(laplacian_pyramid)):
            size = (laplacian_pyramid[i].shape[1],
                    laplacian_pyramid[i].shape[0])
            res = cv2.pyrUp(res, dstsize=size)
            res = cv2.add(res, laplacian_pyramid[i])

        return res
