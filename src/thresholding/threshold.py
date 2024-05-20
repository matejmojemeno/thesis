import glob

import cv2
import numpy as np


def contour_position(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return 0, 0


def draw_contour(contour):
    return cv2.drawContours(
        np.zeros((300, 300), dtype=np.uint8),
        [contour],
        -1,
        color=(255, 255, 255),
        thickness=cv2.FILLED,
    )


def find_contours(img_bin, min_area=500):
    mode = cv2.RETR_LIST
    contours, _ = cv2.findContours(img_bin, mode, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > min_area]


def segmment_sperm(frame):
    contours = find_contours(frame)
    closest_contour = None
    closest_distance = 100000

    if not contours:
        return None

    for contour in contours:
        cx, cy = contour_position(contour)
        distance = np.sqrt((cx - 150) ** 2 + (cy - 150) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_contour = contour

    return draw_contour(closest_contour)


def threshold(frame, thresh):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    _, frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)

    frame //= 255
    frame = 1 - frame

    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, np.ones((3, 3)))
    image_foreground_preprocessed = cv2.erode(
        frame, np.ones((3, 3), np.uint8), iterations=1
    )

    return segmment_sperm(image_foreground_preprocessed)


def segment_video(video_path, out_video_name, thresh):
    video = cv2.VideoCapture(video_path)
    video_output = cv2.VideoWriter(
        f"data/segmented_videos/{out_video_name}",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (300, 300),
        isColor=False,
    )

    success, frame = video.read()
    while success:
        segmented = threshold(frame, thresh)
        if segmented is None:
            return False
        video_output.write(segmented)
        success, frame = video.read()

    video.release()
    video_output.release()
    return True


def main():
    video_paths = glob.glob("data/raw_videos/*.mp4")

    for video_path in video_paths:
        thresh = 85
        success = False
        out_video_name = video_path.split("/")[-1][:-4] + ".mp4"
        while not success:
            success = segment_video(video_path, out_video_name, thresh)
            thresh += 1
        print(thresh)
        segment_video(video_path, out_video_name, thresh + 4)


if __name__ == "__main__":
    main()
