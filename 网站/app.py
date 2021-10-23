#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2019-11-08

import os
from flask import render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ipromoter_mail import *
import data_preprocess
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(1)
app = Flask(__name__)


# 跳转到网页主界面
@app.route('/', methods=['POST', 'GET'])
def Pse_home():
    return render_template('ipromoter_5mC_home.html')


# 跳转到实例界面
@app.route('/Pse_example', methods=['POST', 'GET'])
def Pse_example():
    return render_template('ipromoter_5mC_example.html')


# 跳转到帮助界面
@app.route('/Pse_help', methods=['POST', 'GET'])
def Pse_help():
    return render_template('ipromoter_5mC_help.html')


# 跳转到等待界面
@app.route('/Pse_backhome', methods=['POST', 'GET'])
def Pse_backhome():
    return render_template('ipromoter_5mC_backhome.html')


# 上传fasta文件进行预测
@app.route('/Pse_upload_file', methods=['POST', 'GET'])
def Pse_upload_file():
    if request.method == 'POST':
        f = request.files['file']
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(base_path,
                                   'save_model/funsion_process',
                                   secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        Seq_strand = request.form.get('strand')
        Program_names = request.values.get('Program_name')
        Email_address = request.values.get('Email_address')
        print(Program_names, Email_address)
        print(upload_path)
        executor.submit(ipromoter, Seq_strand, Program_names, upload_path, Email_address)
        return render_template('ipromoter_5mC_backhome.html')
    else:
        return render_template('upload_file.html')


@app.route('/get_requests', methods=['POST', 'GET'])
def ipromoter(Seq_strand, Program_name, upload_path, Email_address):
    Seq_strand = Seq_strand
    Program_name = Program_name
    Email_address = Email_address
    print('1')
    result_load, original_file_path = data_preprocess.getfasta_file(Seq_strand, Program_name, upload_path)
    i_sendmail(Program_name, Email_address, result_load, original_file_path)
    print("Mail send!")


@app.route("/download")
def Pse_download():
    return render_template('ipromoter_5mC_download.html')


@app.route("/download_all")
def download_all():
    return send_from_directory(r"iPromoter-5mC/static/download", filename="all_data.rar",
                               as_attachment=True)


@app.route("/download_benchmark")
def download_benchmark():
    return send_from_directory(r"iPromoter-5mC/static/download", filename="benchmark.rar",
                               as_attachment=True)


@app.route("/download_train_data")
def download_train_data():
    return send_from_directory(r"iPromoter-5mC/static/download",
                               filename="train_data.rar",
                               as_attachment=True)


@app.route("/download_test_data")
def download_test_data():
    return send_from_directory(r"iPromoter-5mC/static/download", filename="test_data.rar",
                               as_attachment=True)


@app.route("/download_stand_alone_software")
def download_stand_alone_data():
    return send_from_directory(r"iPromoter-5mC/static/download",
                               filename="iPromoter-5mC_stand_alone_software.rar",
                               as_attachment=True)

@app.route("/download_Supplementary_experimental_data")
def download_Supplementary_experimental_data():
    return send_from_directory(r"iPromoter-5mC/static/download",
                               filename="Supplementary_experimental_data(liver_cancer).rar",
                               as_attachment=True)


@app.errorhandler(404)
def Pse_page_not_found(e):
    return render_template('ipromoter_5mC_404.html')


if __name__ == "__main__":
    app.run(debug=True)
