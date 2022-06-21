import cv2 as cv
import numpy as np
from statistics import mean
from math import sqrt
from math import ceil
from math import floor
import operator


hsv = None

filename = ""


def distance(p1, p2):
    return sqrt(((int(p1[0]) - int(p2[0])) ** 2) + ((int(p1[1]) - int(p2[1])) ** 2))


def mask_frame(lower_bound, upper_bound, img):
    mask = cv.inRange(img, lower_bound, upper_bound)  # image of the lane lines
    res = cv.bitwise_and(img, img, mask=mask)

    return res


def main():
    global hsv

    cap = cv.VideoCapture(filename)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)
    duration = frame_count / fps
    bins = []

    for i in range(ceil(duration / 82.5)):
        bins.append([])

    frame_count = 0
    total_distance = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if ret:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            sensitivity = 25
            lower_white = np.array([0, 0, 255 - sensitivity])
            upper_white = np.array([255, sensitivity, 255])

            lower_orange = np.array([100, 150, 0])
            upper_orange = np.array([140, 255, 255])

            res_lane = mask_frame(lower_white, upper_white, hsv)
            res_car = mask_frame(lower_orange, upper_orange, hsv)

            loc = np.where(res_car != [0, 0, 0])
            coordinates = set()
            for i in range(loc[0].shape[0]):
                coordinates.add((loc[0][i], loc[1][i]))
            if len(coordinates) == 0:
                continue
            avg = (0, 0)
            for a, b in coordinates:
                avg = (avg[0] + a, avg[1] + b)
            avg = (int(avg[0] / len(coordinates)), int(avg[1] / len(coordinates)))

            line_detect = cv.cvtColor(res_lane, cv.COLOR_HSV2RGB)
            line_detect = cv.cvtColor(line_detect, cv.COLOR_RGB2GRAY)
            blur_gray = cv.GaussianBlur(line_detect, (5, 5), 0)
            edges = cv.Canny(blur_gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            close = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

            line_img = np.copy(close)
            lines = cv.HoughLinesP(close, 1, np.pi / 180, 15, np.array([]), 50, 20)

            skip_frame = False

            for i in range(2):
                lines = cv.HoughLinesP(line_img, 1, np.pi / 180, 15, np.array([]), 50, 20)
                try:
                    for line in lines:
                        l = line[0]
                        cv.line(line_img, (l[0], l[1]), (l[2], l[3]), 255, 3, cv.LINE_AA)
                except TypeError:
                    print('Unusable Frame %s' % frame_count)
                    skip_frame = True
                    break

                line_img = cv.GaussianBlur(line_img, (45, 45), 0)
            if skip_frame:
                continue

            slope = []
            for line in lines:
                l = line[0]
                slope.append((l[3] - l[1]) / (l[2] - l[0]))
            avg_slope = mean(slope)

            centroid_points = []

            ret, thresh = cv.threshold(line_img, 127, 255, 0)
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            try:
                for c in contours:
                    M = cv.moments(c)
                    # cv.imwrite('Here.jpg', frame)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroid_points.append((cX, cY))
            except ZeroDivisionError:
                print("Removing Frame %s" % frame_count)
                continue

            centroids_to_use = [0, 1]
            if len(centroid_points) > 2:
                distances = []
                for i in range(len(centroid_points) - 1):
                    for j in range(i + 1, len(centroid_points)):
                        distances += [(distance(centroid_points[i], centroid_points[j]), [i, j])]
                centroids_to_use = min(distances, key=operator.itemgetter(0))[1]

            try:
                center_of_centroids = (
                    (centroid_points[centroids_to_use[0]][0] + centroid_points[centroids_to_use[1]][0]) / 2,
                    (centroid_points[centroids_to_use[0]][1] + centroid_points[centroids_to_use[1]][1]) / 2)
            except IndexError:
                print("Removing Frame %s" % frame_count)
                continue

            coef_matrix = np.array([-avg_slope, 1, -center_of_centroids[1] + (avg_slope * center_of_centroids[0])])

            left_vertical = np.array([1, 0, 0])
            right_vertical = np.array([1, 0, -line_img.shape[1]])

            inter_left = np.cross(left_vertical, coef_matrix)
            inter_right = np.cross(right_vertical, coef_matrix)

            try:
                cv.line(line_img, (int(inter_left[0]), int(inter_left[1])), (int(inter_right[0]), int(inter_right[1])),
                        255,
                        3, cv.LINE_AA)
            except ValueError:
                print("Removing Frame %s" % frame_count)
                continue

            left_point = np.asarray((int(inter_left[0]), int(inter_left[1])))
            right_point = np.asarray((int(inter_right[0]), int(inter_right[1])))
            car_point = np.asarray(avg[::-1])

            lane_distance = np.linalg.norm(
                np.cross(right_point - left_point, left_point - car_point)) / np.linalg.norm(
                right_point - left_point)
            # print(lane_distance, frame_count)
            if frame_count % 500 == 0:
                print(frame_count)
            total_distance += lane_distance
            bins[floor((frame_count / fps) / 82.5)].append(lane_distance)
        else:
            break
    bins_averaged = []
    for i in bins:
        bins_averaged.append(sum(i) / len(i))
    print()
    print("Bins:", bins_averaged)
    print("Total Distance: %s" % total_distance)


if __name__ == "__main__":
    main()
