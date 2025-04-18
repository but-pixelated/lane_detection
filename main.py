"""
Lane Lines Detection Pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH

Options:
    -h --help                               Show this screen.
    --video                                 Process video file instead of an image.
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from moviepy.video.io.VideoFileClip import VideoFileClip
from CameraCalibration import CameraCalibration
from Thresholding import Thresholding
from PerspectiveTransformation import PerspectiveTransformation
from LaneLines import LaneLines



class FindLaneLines:
    """
    This class is for parameter tuning and processing video/images
    for lane line detection.
    """

    def __init__(self):
        """Initialize application components."""
        self.calibration = CameraCalibration('camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        
    def forward(self, img):
        """
        Process a single frame/image for lane line detection.

        Args:
            img (ndarray): Input image.

        Returns:
            ndarray: Processed image with lane lines overlay.
        """
        
        
        
        out_img = np.copy(img)
        img = self.calibration.undistort(img)
        img = self.transform.forward(img)
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        return out_img
    
    def process_image(self, input_path, output_path):
        """
        Process a single image and save the output.

        Args:
            input_path (str): Path to the input image.
            output_path (str): Path to save the output image.
        """
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)

    def process_video(self, input_path, output_path):
        """
        Process a video frame-by-frame and save the output.

        Args:
            input_path (str): Path to the input video.
            output_path (str): Path to save the output video.
        """
        clip = VideoFileClip(input_path)

        try:
            # Use fl_image for video processing if supported
            out_clip = clip.fl_image(self.forward)
            out_clip.write_videofile(output_path, audio=False)
        except AttributeError as e:
            # Fallback: Process frame-by-frame manually
            print(f"Error using fl_image: {e}. Processing manually.")
            frames = [self.forward(frame) for frame in clip.iter_frames()]
            fps = clip.fps
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()


def main():
    """
    Entry point for the script.
    Parses arguments and processes video or image based on input.
    """
    args = docopt(__doc__)
    input_path = args['INPUT_PATH']
    output_path = args['OUTPUT_PATH']

    find_lane_lines = FindLaneLines()

    if args['--video']:
        find_lane_lines.process_video(input_path, output_path)
    else:
        find_lane_lines.process_image(input_path, output_path)


if __name__ == "__main__":
    main()