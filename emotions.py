from statistics import mode

import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
from keras.models import load_model
from scipy.spatial import distance as dist
from tqdm import tqdm

from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from utils.inference import draw_bounding_box
from utils.inference import draw_text

points = []
FACE = []
EMOTIONS = []
STRESS = []
FACE_DETECT = []
LEFT_EXPRESSION = []
RIGHT_EXPRESSION = []


def writeFile(is_face, face_cord, emotion, stress_level, expression):
    df = pd.DataFrame(
        {'is-face': is_face, 'face-cord': face_cord, 'emotions': emotion,
         'stress-level': stress_level, 'expression': expression})
    # df.to_csv('output.csv', index=False, header=True)
    return df.to_json()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def eye_brow_distance(leye, reye):
    global points
    distq = dist.euclidean(leye, reye)
    points.append(int(distq))
    return distq


def normalize_values(points, disp):
    # global  points
    normalized_value = abs(disp - np.min(points)) / abs(np.max(points) - np.min(points))
    stress_value = np.exp(-normalized_value)
    # print(stress_value)
    if stress_value >= 0.75:
        return stress_value, "High_Stress"
    elif 0.45 <= stress_value < 0.75:
        return stress_value, "Normal"
    else:
        return stress_value, "low_stress"


#
# def emotion_recognition(image):
#     global points
#     face_status=[]
#     face_cords = []
#     emotion = []
#     stress_level = []
#     face_status = []
#     left = []
#     EYE_BLINK_THRESHOLD = 0.2
#     LIPS_DISTANCE_THRESHOLD = 100.0
#     IS_LIP = True
#     emotion_model_path = './models/emotion_model.hdf5'
#     emotion_labels = get_labels('fer2013')
#     frame_window = 10
#     emotion_offsets = (20, 40)
#     detector = dlib.get_frontal_face_detector()
#     emotion_classifier = load_model(emotion_model_path)
#
#     predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
#     emotion_target_size = emotion_classifier.input_shape[1:3]
#     emotion_window = []
#     # cv2.namedWindow('window_frame')
#
#     # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     blink = 0
#     print('Total Frames : {}'.format(total_frames))
#     # is_face = False
#     pbar = tqdm(total=total_frames)
#     while cap.isOpened():  # True:
#         ret, bgr_image = cap.read()
#
#         # cv2.imshow('window_frame', bgr_image)
#         # cv2.waitKey(30)
#         if not ret:
#             break
#         gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#         rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#
#         faces = detector(rgb_image)
#         for face_coordinates in faces:
#             face_cords = []
#             emotion = []
#             stress_level = []
#             face_status = []
#             left = []
#             right = []
#
#             is_face = True
#             face_status.append(is_face)
#             shape = predictor(bgr_image, face_coordinates)
#             x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
#             face_cords.append([x1, x2, y1, y2])
#             (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
#             (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
#             (elStart, elEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#             (erStart, erEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#             (imstart, imend) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
#             (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
#
#             gray_face = gray_image[y1:y2, x1:x2]
#
#             shape = face_utils.shape_to_np(shape)
#
#             try:
#                 gray_face = cv2.resize(gray_face, emotion_target_size)
#                 leyebrow = shape[lBegin:lEnd]
#                 reyebrow = shape[rBegin:rEnd]
#
#                 eleye = shape[elStart:elEnd]
#                 ereye = shape[erStart:erEnd]
#
#                 imstarteye = shape[imstart: imend]
#                 mstarteye = shape[mstart: mend]
#
#                 innerlipshull = cv2.convexHull(imstarteye)
#                 outerlipshull = cv2.convexHull(mstarteye)
#
#                 reyebrowhull = cv2.convexHull(reyebrow)
#                 leyebrowhull = cv2.convexHull(leyebrow)
#
#                 ereyehull = cv2.convexHull(ereye)
#                 eleyehull = cv2.convexHull(eleye)
#
#                 imhull = cv2.convexHull(ereye)
#                 mhull = cv2.convexHull(eleye)
#
#                 leftEAR = eye_aspect_ratio(eleye)
#                 rightEAR = eye_aspect_ratio(ereye)
#
#
#                 # average the eye aspect ratio together for both eyes
#                 ear = (leftEAR + rightEAR) / 2.0
#             except Exception as e:
#                 print('Un handled exception occurred  : ', e)
#                 continue
#             gray_face = preprocess_input(gray_face, True)
#             gray_face = np.expand_dims(gray_face, 0)
#             gray_face = np.expand_dims(gray_face, -1)
#             emotion_prediction = emotion_classifier.predict(gray_face)
#             emotion_probability = np.max(emotion_prediction)
#             emotion_label_arg = np.argmax(emotion_prediction)
#             emotion_text = emotion_labels[emotion_label_arg]
#             emotion.append(emotion_text)
#             emotion_window.append(emotion_text)
#
#             if len(emotion_window) > frame_window:
#                 emotion_window.pop(0)
#             try:
#                 emotion_mode = mode(emotion_window)
#             except Exception as e:
#                 print(e)
#                 continue
#             if emotion_text == 'angry':
#                 color = emotion_probability * np.asarray((255, 0, 0))
#             elif emotion_text == 'sad':
#                 color = emotion_probability * np.asarray((0, 0, 255))
#             elif emotion_text == 'happy':
#                 color = emotion_probability * np.asarray((255, 255, 0))
#             elif emotion_text == 'surprise':
#                 color = emotion_probability * np.asarray((0, 255, 255))
#             else:
#                 color = emotion_probability * np.asarray((0, 255, 0))
#             color = color.astype(int)
#             color = color.tolist()
#
#             draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
#             draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
#                       color, 0, -45, 1, 1)
#
#             bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#             try:
#                 distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
#                 centerOfLeftEyeBrow = leyebrow[2]
#                 centerOfRightEyeBrow = reyebrow[2]
#
#                 midLeftEye = ereye[2]
#                 midRightEye = eleye[2]
#
#                 cv2.circle(bgr_image, (imstarteye[0][0], imstarteye[0][1]), 5, (0, 255, 0), 3)
#                 cv2.circle(bgr_image, (imstarteye[4][0], imstarteye[4][1]), 5, (0, 255, 0), 3)
#
#                 cv2.circle(bgr_image, (centerOfLeftEyeBrow[0], centerOfLeftEyeBrow[1]), 5, (0, 0, 0), 3)
#                 cv2.circle(bgr_image, (midLeftEye[0], midLeftEye[1]), 5, (255, 0, 0), 3)
#                 # #
#                 cv2.circle(bgr_image, (centerOfRightEyeBrow[0], centerOfRightEyeBrow[1]), 5, (0, 0, 255), 3)
#                 cv2.circle(bgr_image, (midRightEye[0], midRightEye[1]), 5, (0, 0, 255), 3)
#
#                 lips_distance = (dist.euclidean(imstarteye[4], imstarteye[0]))
#                 if IS_LIP:
#                     LIPS_DISTANCE_THRESHOLD = lips_distance - ((lips_distance/100) * 20)
#                     IS_LIP = False
#
#                 expression_level = (dist.euclidean(centerOfLeftEyeBrow, midLeftEye) + dist.euclidean(
#                     centerOfRightEyeBrow, midRightEye)) / 2
#                 left.append(expression_level)
#                 cv2.putText(bgr_image, "Expression level:{}".format(str(int(expression_level))), (20, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (0, 0, 0), 2)
#                 cv2.putText(bgr_image, "Emotion prob:{}".format(str(int(emotion_probability*100))), (20, 250),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (0, 0, 0), 2)
#
#                 stress_value, stress_label = normalize_values(points, distq)
#                 stress_level.append(stress_value)
#
#                 cv2.drawContours(bgr_image, [reyebrowhull], -1, (0, 255, 0), 1)
#                 cv2.drawContours(bgr_image, [leyebrowhull], -1, (0, 255, 0), 1)
#                 cv2.drawContours(bgr_image, [ereyehull], -1, (0, 255, 55), 1)
#                 cv2.drawContours(bgr_image, [eleyehull], -1, (0, 255, 55), 1)
#                 cv2.drawContours(bgr_image, [innerlipshull], -1, (0, 255, 55), 1)
#                 # cv2.drawContours(bgr_image, [outerlipshull], -1, (0, 255, 55), 1)
#                 cv2.putText(bgr_image, "{}:{}".format(stress_label, str(int(stress_value * 100))), (20, 40),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1, (0, 0, 0), 2)
#                 if lips_distance > LIPS_DISTANCE_THRESHOLD:
#                     cv2.putText(bgr_image, "{}_{}".format('Smiled', LIPS_DISTANCE_THRESHOLD), (20, 180),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, (0, 0, 0), 2)
#                 if ear < EYE_BLINK_THRESHOLD:
#                     blink += 1
#                     cv2.putText(bgr_image, "{}".format('Eye blinked'), (20, 80),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, (0, 0, 0), 2)
#                 else:
#                     cv2.putText(bgr_image, "{}".format('Eyes Open'), (20, 80),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 1, (0, 0, 0), 2)
#             except Exception as e:
#                 print('Unhandled exception occurred : {}'.format(e))
#             bgr_image = cv2.resize(bgr_image, (940, 840))
#
#             cv2.imshow('window_frame', bgr_image)
#
#             # print(is_face,face_cords,emotion,stress_level)
#             #
#             if cv2.waitKey(30) & 0xFF == ord('q'):
#                 break
#         FACE.append(face_status)
#         EMOTIONS.append(emotion)
#         STRESS.append(stress_level)
#         FACE_DETECT.append(face_cords)
#         LEFT_EXPRESSION.append(left)
#         pbar.update(1)
#
#     df = writeFile(FACE, FACE_DETECT, EMOTIONS, STRESS, LEFT_EXPRESSION)
#     cap.release()
#     # return df
#     cv2.destroyAllWindows()

class emotion_analyzer():

    def __init__(self):
        emotion_model_path = './models/emotion_model.hdf5'
        self.emotion_labels = get_labels('fer2013')
        self.frame_window = 10
        self.emotion_offsets = (20, 40)
        self.detector = dlib.get_frontal_face_detector()
        self.emotion_classifier = load_model(emotion_model_path)
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]

    def emotion_recognition(self, bgr_image, lipdistance, eyeblink, expressionthreshold,):
        EYE_BLINK_THRESHOLD = eyeblink
        LIPS_DISTANCE_THRESHOLD = lipdistance
        EXPRESSION_THRESHOLD = expressionthreshold
        IS_LIP = True
        emotion_window = []
        blink = 0
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = self.detector(rgb_image)
        for face_coordinates in faces:
            _, _, w, h = face_utils.rect_to_bb(face_coordinates)

            areaofface = w * h
            if areaofface <= 15000:
                bgr_image = cv2.resize(bgr_image, (720, 640))
                return bgr_image, areaofface
            
            face_cords = []
            emotion = []
            stress_level = []
            face_status = []
            left = []
            right = []

            is_face = True
            face_status.append(is_face)
            shape = self.predictor(bgr_image, face_coordinates)
            x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), self.emotion_offsets)
            face_cords.append([x1, x2, y1, y2])
            (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
            (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
            (elStart, elEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (erStart, erEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            (imstart, imend) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
            (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

            gray_face = gray_image[y1:y2, x1:x2]

            shape = face_utils.shape_to_np(shape)

            try:
                gray_face = cv2.resize(gray_face, self.emotion_target_size)
                leyebrow = shape[lBegin:lEnd]
                reyebrow = shape[rBegin:rEnd]

                eleye = shape[elStart:elEnd]
                ereye = shape[erStart:erEnd]

                imstarteye = shape[imstart: imend]
                mstarteye = shape[mstart: mend]

                innerlipshull = cv2.convexHull(imstarteye)
                outerlipshull = cv2.convexHull(mstarteye)

                reyebrowhull = cv2.convexHull(reyebrow)
                leyebrowhull = cv2.convexHull(leyebrow)

                ereyehull = cv2.convexHull(ereye)
                eleyehull = cv2.convexHull(eleye)

                imhull = cv2.convexHull(ereye)
                mhull = cv2.convexHull(eleye)

                leftEAR = eye_aspect_ratio(eleye)
                rightEAR = eye_aspect_ratio(ereye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
            except Exception as e:
                print('Un handled exception occurred  : ', e)
                continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            emotion.append(emotion_text)
            emotion_window.append(emotion_text)

            if len(emotion_window) > self.frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except Exception as e:
                print(e)
                continue
            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))
            color = color.astype(int)
            color = color.tolist()


            draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
            draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            try:
                distq = eye_brow_distance(leyebrow[-1], reyebrow[0])
                centerOfLeftEyeBrow = leyebrow[2]
                centerOfRightEyeBrow = reyebrow[2]

                midLeftEye = ereye[2]
                midRightEye = eleye[2]

                cv2.circle(bgr_image, (imstarteye[0][0], imstarteye[0][1]), 5, (0, 255, 0), 3)
                cv2.circle(bgr_image, (imstarteye[4][0], imstarteye[4][1]), 5, (0, 255, 0), 3)

                cv2.circle(bgr_image, (centerOfLeftEyeBrow[0], centerOfLeftEyeBrow[1]), 5, (0, 0, 0), 3)
                cv2.circle(bgr_image, (midLeftEye[0], midLeftEye[1]), 5, (255, 0, 0), 3)
                # #
                cv2.circle(bgr_image, (centerOfRightEyeBrow[0], centerOfRightEyeBrow[1]), 5, (0, 0, 255), 3)
                cv2.circle(bgr_image, (midRightEye[0], midRightEye[1]), 5, (0, 0, 255), 3)

                lips_distance = (dist.euclidean(imstarteye[4], imstarteye[0]))
                # if IS_LIP:
                #     LIPS_DISTANCE_THRESHOLD = lips_distance - ((lips_distance / 100) * 20)
                #     IS_LIP = False

                expression_level = (dist.euclidean(centerOfLeftEyeBrow, midLeftEye) + dist.euclidean(
                    centerOfRightEyeBrow, midRightEye)) / 2


                left.append(expression_level)
                if expression_level > EXPRESSION_THRESHOLD:
                    cv2.putText(bgr_image, "Expression level:{}".format(str(int(expression_level))), (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 2)
                cv2.putText(bgr_image, "Emotion prob:{}".format(str(int(emotion_probability * 100))), (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                cv2.putText(bgr_image, "Lip Distance:{}".format(str(int(LIPS_DISTANCE_THRESHOLD))), (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                cv2.putText(bgr_image, "EYE Distance:{}".format(str(float(EYE_BLINK_THRESHOLD))), (20, 260),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                cv2.putText(bgr_image, "Expression Threshold:{}".format(str(float(EXPRESSION_THRESHOLD))), (20, 320),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                cv2.putText(bgr_image, "Area of Face:{}".format(str(float(areaofface))), (20, 360),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                stress_value, stress_label = normalize_values(points, distq)
                stress_level.append(stress_value)

                cv2.drawContours(bgr_image, [reyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(bgr_image, [leyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(bgr_image, [ereyehull], -1, (0, 255, 55), 1)
                cv2.drawContours(bgr_image, [eleyehull], -1, (0, 255, 55), 1)
                cv2.drawContours(bgr_image, [innerlipshull], -1, (0, 255, 55), 1)
                # cv2.drawContours(bgr_image, [outerlipshull], -1, (0, 255, 55), 1)
                cv2.putText(bgr_image, "{}:{}".format(stress_label, str(int(stress_value * 100))), (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                if lips_distance > LIPS_DISTANCE_THRESHOLD:
                    cv2.putText(bgr_image, "{}: {}".format('Person Smiled', lipdistance), (20, 300),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 2)
                if ear < EYE_BLINK_THRESHOLD:
                    blink += 1
                    cv2.putText(bgr_image, "{}".format('Eye blinked'), (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 2)
                else:
                    cv2.putText(bgr_image, "{}".format('Eyes Open'), (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 2)
            except Exception as e:
                print('Unhandled exception occurred : {}'.format(e))
            bgr_image = cv2.resize(bgr_image, (720, 640))
            return bgr_image, areaofface
# df= emotion_recognition('c.mov')
# print(df)
