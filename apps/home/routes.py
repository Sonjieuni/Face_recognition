# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
from apps import upload_dir
from flask import render_template, request, flash, redirect, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound
from apps.home import blueprint
from .forms import FileForm


@blueprint.route('/')
def route_default():
    return render_template('home/choice.html', segment='choice')


@blueprint.route('/Capture')
def capture():
    return render_template('home/Capture.html', segment='choice')


@blueprint.route('/index')
@login_required
def index():
    return render_template('home/AdminMain.html', segment='index')


# 얼굴 인식(이름 목록)
@blueprint.route("/name_list", methods=["GET", "POST"])
@login_required
def name_list():
    # 사용자 이름 디렉터리 확인 후 없으면 생성
    face_names = os.path.join(upload_dir, "face_names")
    if not os.path.exists(face_names):
        os.makedirs(face_names)

    form = FileForm(request.form)
    name = form.name.data
    id = form.id.data
    messages = []

    if request.method == "POST":
        if name:
            # 디렉터리 생성
            name_dir = os.path.join(upload_dir, "face_names", name, id)
            if not os.path.exists(name_dir):
                os.makedirs(name_dir)
            messages.append("데이터가 등록되었습니다.")

    # 사용자 이름 불러오기
    names = os.listdir(face_names)

    name_list = []
    for name in names:
        name_path = os.path.join(upload_dir, "face_names", name)
        files = os.listdir(name_path)
        count = len(files)
        data = {"name": name, "count": count}
        name_list.append(data)

    return render_template("home/name_list.html", name_list=name_list, messages=messages, form=form)


# 이름 삭제
@blueprint.route("/name_delete/<string:name>")
@login_required
def name_delete(name):
    name_path = os.path.join(upload_dir, "face_names", name)
    if os.path.exists(name_path):
        import shutil
        shutil.rmtree(name_path)
    flash("데이터가 삭제되었습니다.")
    return redirect(url_for("Capture_blueprint.name_list"))


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
