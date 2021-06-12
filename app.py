# # from flask import Flask, redirect, url_for, request, jsonify
#from emotions import emotion_recognition
# # import os
#
# df = emotion_recognition(0)
#
# # app = Flask(__name__)
#
#
# # @app.route('/')
# # def success():
# #     return 'Server Running'
#
#
# # @app.route('/api/v1/emotion', methods=['POST'])
# # def emotion_detector():
# #     if request.method == 'POST':
# #         rqst = request.get_json()
# #         print(rqst)
# #         file_id = rqst.get("file_id")
# #         file_name = rqst.get("file_name")
# #         disk_file_name = rqst.get("disk_file_name")
# #         file_path = rqst.get("file_path")
# #         content_type = rqst.get("content_type")
# #         content_length = rqst.get("content_length")
# #         content = rqst.get("content")
# #         df = emotion_recognition(os.path.join(file_path, file_name))
# #         # resp = jsonify(success=True)
# #         return df
# #     else:
# #         resp = jsonify(success=False)
# #     return resp
#
#
# # if __name__ == '__main__':
# #     app.run(debug=True)
#
import cv2
import tkinter as tk
from PIL import Image, ImageTk

import emotions as em

ea = em.emotion_analyzer()


class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self._set_lipdistance(60)
        self._set_eyedistance(0.2)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 20  # Interval in ms to get the latest frame
        # Create canvas for image

        self.canvas = tk.Canvas(self.window, width=640, height=480)
        self.canvas.grid(row=0, column=0)
        # Update image on canvas
        self.update_image()

        self.lipdistance_label = tk.Label(window, text='Distance between lips (0-100)')
        self.canvas.create_window(100, 15, window=self.lipdistance_label)
        self.lipdistance_entry = tk.Entry(window)
        self.canvas.create_window(100, 50, window=self.lipdistance_entry)

        self.eyeblink_label = tk.Label(window, text='Distance between Eyes (0 to 1)')
        self.canvas.create_window(310, 15, window=self.eyeblink_label)
        self.eyeblink_entry = tk.Entry(window)
        self.canvas.create_window(310, 50, window=self.eyeblink_entry)

        self.updatethresh_btn = tk.Button(text='Update Threshold', bg='brown', command=self.updatethresh, fg='white',
                                          font=('helvetica', 9, 'bold'))

        self.canvas.create_window(450, 50, window=self.updatethresh_btn)


    def updatethresh(self):
        self._set_eyedistance(self.eyeblink_entry.get())
        self._set_lipdistance(self.lipdistance_entry.get())
        self.update_image()
        # self.cap.release()
        # self.window.destroy()
        # _start()

    def _set_eyedistance(self, v):
        self.eye_distance = float(v)

    def _set_lipdistance(self, v):
        self.lip_distance = float(v)

    def _get_eyedistance(self):
        return self.eye_distance

    def _get_lipdistance(self):
        return self.lip_distance

    def update_image(self):
        # Get the latest frame and convert image format
        ld = self._get_lipdistance()
        ed = self._get_eyedistance()
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # to RGB
        self.image = ea.emotion_recognition(self.image, ld, ed)
        # print(type(self.image))
        # if self.image is not None:
        self.image = Image.fromarray(self.image)  # to PIL format
        self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format
        # Update image
        self.canvas.create_image(0, 100, anchor=tk.NW, image=self.image)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)


def _start():

    root = tk.Tk()
    m = MainWindow(root, cv2.VideoCapture(0))
    # MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()


if __name__ == "__main__":
    _start()
    # print(self.lipdistance_entry.get())
