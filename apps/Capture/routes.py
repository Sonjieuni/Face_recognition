import base64
import json
import os
import pickle
import uuid

import cv2
import numpy as np
from flask import render_template, request, jsonify, url_for, redirect, flash
from flask_login import login_required

from apps import upload_dir, models_dir, face_names_dir, fonts_dir
from apps.Capture import blueprint
from .forms import FileForm


# 분류기
detector = cv2.CascadeClassifier(os.path.join(models_dir, "haarcascades", "haarcascade_frontalface_default.xml"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def on_json_loading_failed_return_dict(e):
    return {}


# 얼굴 감지(backend)
@blueprint.route("/detect", methods=["POST"])
@login_required
def detect_post():
    request.on_json_loading_failed = on_json_loading_failed_return_dict
    if request.get_json()["imageData"]:
        # base64 데이터를 OpenCV 이미지 데이터로 변환하기
        im_bytes = base64.b64decode(request.get_json()["imageData"])
        img_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)

        # 흑백처리(BGR -> GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 찾기
        try:
            faces = detector.detectMultiScale(img_gray)
            print(faces)

            # 찾은 얼굴
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        except:
            pass

        # 이미지 파일을 base64로 인코딩
        img_jpg = cv2.imencode(".jpg", img)
        image_base64 = base64.b64encode(img_jpg[1]).decode("utf-8")

        return jsonify(image_base64=image_base64)
    return jsonify()


# 얼굴 감지(frontend)
@blueprint.route("/detect", methods=["GET"])
@login_required
def detect():
    return render_template("face/detect.html")


# 실시간 얼굴 감지(frontend)
@blueprint.route("/Capture", methods=["GET"])
@login_required
def detect_realtime():
    return render_template("face/Capture.html")

# 이름 삭제
@blueprint.route("/name_delete/<string:name>")
@login_required
def name_delete(name):
    name_path = os.path.join(upload_dir, "face_names", name)
    if os.path.exists(name_path):
        import shutil
        shutil.rmtree(name_path)
    flash("데이터가 삭제되었습니다.")
    return redirect(url_for("face_blueprint.name_list"))


# 얼굴 인식(이름에 해당하는 사진 목록)
@blueprint.route("/face_list/<string:name>", methods=["GET", "POST"])
@login_required
def face_list(name):
    form = FileForm(request.form)
    messages = []
    errors = []

    # 사진 등록
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_image_file(file.filename):
            # 확장자
            file_extension = file.filename.split(".")[-1]

            # 파일 저장
            new_filename = str(uuid.uuid4()) + "." + file_extension
            file.save(os.path.join(upload_dir, "face_names", name, new_filename))
            messages.append("데이터가 등록되었습니다.")

    # 사진 파일 불러오기
    faces = os.listdir(os.path.join(upload_dir, "face_names", name))

    face_list = []
    for face in faces:
        face_path = os.path.join(upload_dir, "face_names", name, face)

        # 이미지 파일을 base64로 인코딩
        with open(face_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 확장자
        file_extension = face.split(".")[-1]

        # 파일 생성 시간
        from datetime import datetime
        create_unixtime = os.path.getctime(face_path)
        create_timestamp = datetime.fromtimestamp(create_unixtime)
        create_datetime = datetime.strftime(create_timestamp, '%Y-%m-%d %H:%M:%S')

        # 파일 크기
        file_size = os.path.getsize(face_path)

        data = {"name": name, "image_base64": image_base64, "file_extension": file_extension, "file_name": face, "create_datetime": create_datetime, "file_size": file_size}
        face_list.append(data)

    return render_template("home/face_list.html", face_list=face_list, name=name, messages=messages, form=form)


# 사진 등록
@blueprint.route("/face_upload", methods=["POST"])
@login_required
def face_upload():
    form = FileForm(request.form)

    file = request.files["file"]
    name = form.data["name"]

    # 저장 위치
    name_path = os.path.join(upload_dir, "face_names", name)

    # 파일 저장
    new_filename = str(uuid.uuid4()) + ".png"
    file.save(os.path.join(name_path, new_filename))

    return jsonify(result="ok")


# 사진 삭제
@blueprint.route("/face_delete/<string:name>/<string:file_name>")
@login_required
def face_delete(name, file_name):
    file_path = os.path.join(upload_dir, "face_names", name, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    flash("데이터가 삭제되었습니다.")
    return redirect(url_for("face_blueprint.face_list", name=name))


# 사진 학습
@blueprint.route("/train_face", methods=["POST"])
@login_required
def train_face():
    if not os.path.exists(custom_face_model_dir):
        os.makedirs(custom_face_model_dir)

    model_file = os.path.join(custom_face_model_dir, "encodings.pickle")
    if os.path.exists(model_file):
        os.remove(model_file)

    known_face_names = []
    known_face_encodings = []
    known_images_dirs = os.listdir(face_names_dir)

    for known_images_dir in known_images_dirs:
        known_file_path = os.path.join(face_names_dir, known_images_dir)
        known_files = os.listdir(known_file_path)

        for known_file in known_files:

            # 파일명, 확장자
            known_face_names.append(known_images_dir)

            # 인코딩
            known_image_path = os.path.join(known_file_path, known_file)
            known_image = face_recognition.load_image_file(known_image_path)
            known_face_encoding = face_recognition.face_encodings(known_image)[0]
            known_face_encodings.append(known_face_encoding)

    # 인코딩 데이터 저장
    data = {"encodings": known_face_encodings, "names": known_face_names}
    f = open(model_file, "wb")
    f.write(pickle.dumps(data))
    f.close()

    return jsonify(result="ok")


# 얼굴 인식(frontend)
@blueprint.route("/recognize", methods=["GET"])
@login_required
def recognize():
    return render_template("face/recognize.html")


# 실시간 얼굴 인식(frontend)
@blueprint.route("/recognize_realtime", methods=["GET"])
@login_required
def recognize_realtime():
    return render_template("face/recognize_realtime.html")


# 얼굴 인식(backend)
@blueprint.route("/recognize", methods=["POST"])
@login_required
def recognize_post():
    request.on_json_loading_failed = on_json_loading_failed_return_dict
    if request.get_json()["imageData"]:
        # 인코딩 모델 불러오기
        model_file = os.path.join(custom_face_model_dir, "encodings.pickle")
        data = pickle.loads(open(model_file, "rb").read())

        # base64 데이터를 OpenCV 이미지 데이터로 변환하기
        im_bytes = base64.b64decode(request.get_json()["imageData"])
        img_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 얼굴 찾기
        boxes = face_recognition.face_locations(img_rgb)
        encodings = face_recognition.face_encodings(img_rgb, boxes)

        # 얼굴 인식
        who_names = []
        for encoding in encodings:
            distances = face_recognition.face_distance(data["encodings"], encoding)

            name = "몰라"
            if min(distances) < 0.4:
                index = np.argmin(distances)
                name = data["names"][index]
            who_names.append(name)

        # 얼굴 표시
        for (top, right, bottom, left), name in zip(boxes, who_names):
            cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(img_bgr, (left, bottom), (right, bottom + 25), (0, 0, 255), -1)

        # 이름 표시
        from PIL import ImageFont, ImageDraw, Image
        img_pillow = Image.fromarray(img_bgr)
        draw = ImageDraw.Draw(img_pillow)
        font = ImageFont.truetype(os.path.join(fonts_dir, "gulim.ttc"), 14)
        for (top, right, bottom, left), name in zip(boxes, who_names):
            draw.text((left + 10, bottom + 5), name, font=font, fill=(255, 255, 255))

        img_bgr = np.array(img_pillow)

        # 이미지 파일을 base64로 인코딩
        img_jpg = cv2.imencode(".jpg", img_bgr)
        image_base64 = base64.b64encode(img_jpg[1]).decode("utf-8")

        return jsonify(image_base64=image_base64)
    return jsonify()


"""
# 얼굴 인식(backend)
@blueprint.route("/recognize", methods=["POST"])
@login_required
def recognize_post():
    request.on_json_loading_failed = on_json_loading_failed_return_dict
    if request.get_json()["imageData"]:
        currentname = "unknown"

        model_file = os.path.join(custom_face_model_dir, "encodings.pickle")
        data = pickle.loads(open(model_file, "rb").read())

        # base64 데이터를 OpenCV 이미지 데이터로 변환하기
        im_bytes = base64.b64decode(request.get_json()["imageData"])
        img_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 찾기
        boxes = face_recognition.face_locations(img_rgb)
        encodings = face_recognition.face_encodings(img_rgb, boxes)

        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(img, (left, top), (right, bottom),
                          (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (0, 255, 255), 2)

        # 이미지 파일을 base64로 인코딩
        img_jpg = cv2.imencode(".jpg", img)
        image_base64 = base64.b64encode(img_jpg[1]).decode("utf-8")

        return jsonify(image_base64=image_base64)
    return jsonify()
"""


"""
@blueprint.route("/face_recognition", methods=["GET", "POST"])
@login_required
def face_recognition():
    form = FileForm(request.form)
    if request.method == "POST":
        file = request.files["file"]
        if request.method == "POST":
            print("알고있는 대상 인코딩 시작")

            known_face_names = []
            known_face_encodings = []

            # 알고있는 대상
            known_images_dirs = os.listdir(face_images_dir)

            for known_images_dir in known_images_dirs:
                print(known_images_dir)
                known_file_path = os.path.join(face_images_dir, known_images_dir)
                known_files = os.listdir(known_file_path)

                for known_file in known_files:
                    print(known_file)

                    # 파일명, 확장자
                    known_face_names.append(known_images_dir)

                    # 인코딩
                    import face_recognition
                    known_image_path = os.path.join(known_file_path, known_file)
                    known_image = face_recognition.load_image_file(known_image_path)
                    known_face_encoding = face_recognition.face_encodings(known_image)[0]
                    known_face_encodings.append(known_face_encoding)

            print("알고있는 대상 인코딩 완료")

            # 처리할 이미지 임시 저장 위치
            face_temp_path = os.path.join(upload_dir, "face_temp_path")
            if not os.path.exists(face_temp_path):
                os.makedirs(face_temp_path)

            # 파일 저장
            new_filename = str(uuid.uuid4())
            file.save(os.path.join(face_temp_path, new_filename))

            # Read image
            who = face_recognition.load_image_file(os.path.join(face_temp_path, new_filename))
            who_face_locations = face_recognition.face_locations(who)

            if len(who_face_locations) > 0:
                who_face_encodings = face_recognition.face_encodings(who, who_face_locations)

                # 얼굴 인식
                who_face_names = []
                for who_face_encoding in who_face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, who_face_encoding)
                    #print(distances)

                    name = "몰라"
                    if min(distances) < 0.4:
                        index = np.argmin(distances)
                        name = known_face_names[index]
                    who_face_names.append(name)

                who_bgr = cv2.cvtColor(who, cv2.COLOR_RGB2BGR)

                # 얼굴 표시
                for (top, right, bottom, left), name in zip(who_face_locations, who_face_names):
                    cv2.rectangle(who_bgr, (left, top), (right, bottom), (0, 0, 255), 1)
                    cv2.rectangle(who_bgr, (left, bottom), (right, bottom + 25), (0, 0, 255), -1)

                # 이름 표시
                from PIL import ImageFont, ImageDraw, Image
                who_img_pillow = Image.fromarray(who_bgr)
                draw = ImageDraw.Draw(who_img_pillow)
                font = ImageFont.truetype("gulim.ttc", 14)
                for (top, right, bottom, left), name in zip(who_face_locations, who_face_names):
                    draw.text((left + 10, bottom + 5), name, font=font, fill=(255, 255, 255))

                who_img_bgr = np.array(who_img_pillow)

            # 이미지 파일을 base64로 인코딩
            img_jpg = cv2.imencode(".jpg", who_img_bgr)
            image_base64 = base64.b64encode(img_jpg[1]).decode("utf-8")

            # 파일 삭제
            os.remove(os.path.join(face_temp_path, new_filename))

            return jsonify(image_base64=image_base64)
        return jsonify()
    return render_template("face/face_recognition.html", form=form)
"""