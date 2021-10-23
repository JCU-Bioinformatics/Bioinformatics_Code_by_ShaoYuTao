#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2019-11-08

from flask import Flask
from flask_script import Manager
from flask_mail import Mail, Message

app=Flask(__name__)

# 配置邮件：服务器／端口／安全套接字层／邮箱名／授权码/默认的发送人
app.config['MAIL_SERVER'] = "smtp.163.com"
app.config['MAIL_PORT'] = 25
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = "jdz_biosyt@163.com"
# 这里的密码是你在邮箱中的授权码
app.config['MAIL_PASSWORD'] = "SKYMGJJDHVTNLQWQ"
# 显示发送人的名字
app.config['MAIL_DEFAULT_SENDER'] = 'jci<jci_zl@163.com>'
manager = Manager(app)
mail = Mail(app)

def i_sendmail(Program_name, Email_address, result_load, original_file_path):
    # sender 发送方，recipients 邮件接收方列表
    Program_name = Program_name
    Email_address = Email_address
    Original_file_path = original_file_path
    Result_load = result_load
    title_name = "ipromoter-5mC server: " + str(Program_name)
    msg = Message(title_name,sender=app.config['MAIL_DEFAULT_SENDER'],recipients=[Email_address,app.config['MAIL_USERNAME']])
    # msg.body 邮件正文
    msg.body = "ipromoter-5mC server : " + str(Program_name)
    msg.html = '<b>Program_name : ' + str(Program_name) + '</b>' \
                                                          '<p><i>If any question,please send email to jci_zl@163.com!</i></p>'
    # msg.attach 邮件附件添加
    # msg.attach("文件名", "类型", 读取文件）
    # with app.open_resource("test\photo.jpg") as fp:
    #     msg.attach("photo.jpg", "photo/jpg", fp.read())
    with app.open_resource(str(Result_load)) as fp:
        msg.attach(str(Original_file_path), "/csv", fp.read())
        print(Original_file_path, Result_load)

    with app.app_context():
        print('1')
        mail.send(msg)


    print("Mail sent")


if __name__=="__main__":
    manager.run()

