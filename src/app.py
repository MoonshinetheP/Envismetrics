"""
运行逻辑：

1. 使用 JS 背后提交数据。
2. 提交之后展跳转到结果页面，每秒检查一次是否执行完毕。
    /<module>/<version>?step=xx
    /check?module=xx&version=xx&step=xx
3. 后台执行任务，任务完成后会生成结果文件。
    /upload?module=xx&version=xx&step=xx
    background_task()
4. 前台检查执行完毕后，刷新页面，显示结果。

"""

from werkzeug.utils import secure_filename
from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, send_file, abort
import time
import datetime
import logging
import os
import json
import configparser
from datetime import datetime
from HDV import HDV
from CV import CV
import hashlib
import shutil
from config import *
from utils import init_logging, check_folders
import threading
import traceback
from CA import CA
import sys
import os

sys.path.append(os.path.dirname(__file__))
from config import *
from utils import init_logging, check_folders

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

check_folders()
init_logging()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/doc")
def demo():
    notes = []
    return render_template('doc.html', notes=notes)

@app.route("/")
def index():
    data = {}
    return render_template('index.html', data=data)

# Menu 1
@app.route("/hyd_elec")
def hyd_elec():
    return render_template('m1_hyd_elec.html')

@app.route("/hyd_elec/<version>")
def hyd_elec2(version=None):
    data = {
        'version': version
    }
    module = 'HDV'
    step = int(request.args.get('step', '2'))
    step = 2 if step < 2 else step

    data_path = os.path.join('outputs', version)
    if not os.path.exists(data_path):
        abort(404)
    data_file = os.path.join(data_path, 'data.json')

    method = int(request.args.get('method', '-1'))
    if os.path.exists(data_file):
        try:
            data = json.loads(open(data_file).read())
            data = data[module]
            print('---')
            print(data)
            # 检查状态
            if step == 2:
                f = 'form1'
            elif step == 3:
                f = 'form2_{}'.format(method)
            else:
                f = 'form1'

            status = data[f]['status']
        except Exception as e:
            traceback.print_exc()
            data = {}
            status = 'processing'
    else:
        traceback.print_exc()
        data = {}
        status = 'processing'

    data['method'] = method
    data['processing_display'] = 'none' if status == 'done' else 'block'
    data['form1_processing_display'] = 'block' if status == 'done' else 'none'
    data['version'] = version
    data['step'] = step

    if step == 2:
        return render_template('m1_hyd_elec_step2.html', data=data)
    elif step == 3:
        return render_template('m1_hyd_elec_step3.html', data=data)
    else:
        return render_template('m1_hyd_elec_step2.html', data=data)

# Menu 2
@app.route("/cv")
def cv():
    return render_template('m2_cv.html')

@app.route("/cv/<version>")
def cv2(version=None):
    data = {
        'version': version
    }
    module = 'CV'
    step = int(request.args.get('step', '2'))
    step = 2 if step < 2 else step
    data_path = os.path.join('outputs', version)
    if not os.path.exists(data_path):
        abort(404)

    data_file = os.path.join(data_path, 'data.json')

    if os.path.exists(data_file):
        try:
            data = json.loads(open(data_file).read())
            data = data[module]
            print("data:", json.dumps(data))
            # 检查状态
            kk = 'form{}'.format(step - 1)  # Step k 要用到 Form k-1 的结果。
            status = data[kk]['status']
        except Exception as e:
            traceback.print_exc()
            data = {}
            status = 'processing'
    else:
        traceback.print_exc()
        data = {}
        status = 'processing'
    data['status'] = status
    data['processing_display'] = 'none' if status == 'done' else 'block'
    data['version'] = version
    data['step'] = step

    if step == 2:
        return render_template('m2_cv_step2.html', data=data)
    else:
        func = int(request.args.get('func', '-1'))
        if func == 3:
            return render_template('m2_cv_step3_func3.html', data=data)
        elif func == 4:
            return render_template('m2_cv_step3_func4.html', data=data)
        elif func == 5:
            return render_template('m2_cv_step3_func5.html', data=data)
        else:
            return render_template('m2_cv_step3.html', data=data)
    return render_template('m2_cv_step2.html', data=data)

@app.route("/cv/results/<version>")
def cv_res(version=None):
    data = {
        'version': version
    }
    module = 'CV'
    func = int(request.args.get('func', '0'))

    data_path = os.path.join('outputs', version)
    if not os.path.exists(data_path):
        abort(404)

    data_file = os.path.join(data_path, 'data.json')

    if os.path.exists(data_file):
        try:
            data = json.loads(open(data_file).read())
            data = data[module]
            print('---')
            print(data)
            # 检查状态
            kk = 'form{}'.format(func)  # Step k 要用到 Form k-1 的结果。
            status = data[kk]['status']
        except Exception as e:
            traceback.print_exc()
            data = {}
            status = 'processing'
    else:
        traceback.print_exc()
        data = {}
        status = 'processing'

    data['status'] = status
    data['func'] = func
    data['processing_display'] = 'none' if status == 'done' else 'block'
    data['version'] = version

    if func == 3:
        return render_template('m2_cv_step3_func3_res.html', data=data)
    elif func == 4:
        return render_template('m2_cv_step3_func4_res.html', data=data)
    elif func == 5:
        return render_template('m2_cv_step3_func5_res.html', data=data)
    else:
        abort(404)


# Menu 3
@app.route("/step_methods")
def step_methods():
    return render_template('m3_step_methods.html')

@app.route("/step_methods/<version>")
def step_methods2(version=None):
    data = {
        'version': version
    }
    module = 'CA'
    step = int(request.args.get('step', '2'))
    step = 2 if step < 2 else step

    data_path = os.path.join('outputs', version)
    if not os.path.exists(data_path):
        abort(404)
    data_file = os.path.join(data_path, 'data.json')


    if os.path.exists(data_file):
        try:
            data = json.loads(open(data_file).read())
            data = data[module]
            print('---')
            print(data)
            # 检查状态
            f = 'form{}'.format(step - 1)
            status = data[f]['status']
        except Exception as e:
            traceback.print_exc()
            data = {}
            status = 'processing'
    else:
        traceback.print_exc()
        data = {}
        status = 'processing'

    data['processing_display'] = 'none' if status == 'done' else 'block'
    data['form1_processing_display'] = 'block' if status == 'done' else 'none'
    data['version'] = version
    data['step'] = step

    if step == 2:
        return render_template('m3_step_methods_step2.html', data=data)
    elif step == 3:
        return render_template('m3_step_methods_step3.html', data=data)
    else:
        return render_template('m3_step_methods_step2.html', data=data)

@app.route("/check/<module>/<version>")
def check(module, version):
    step = int(request.args.get('step', '1'))
    module = module.upper()

    print("version: {}, module: {}, step: {}".format(version, module, step) )

    data_file = os.path.join('outputs', version, 'data.json')
    if not os.path.exists(data_file):
        data = {'result': 'processing'}
        return jsonify(data)

    data = json.loads(open(data_file).read())

    if module not in data.keys():
        data = {'result': 'processing'}
        return jsonify(data)

    if module.upper() == 'CV':
        func = int(request.args.get('func', '0'))
        if func > 0:
            f = 'form{}'.format(func)
        else:
            f = 'form{}'.format(step-1)

        try:
            if data[module][f]['status'] == 'done':
                # print('11111')
                data = {'result': 'done'}
                return jsonify(data)
            else:
                data = {'result': data[module][f]['status']}
                return jsonify(data)
        except Exception as e:
            # print('22222')
            data = {'result': 'processing'}
            return jsonify(data)
    elif module.upper() == 'HDV':
        if step == 2:
            f = 'form1'
        elif step == 3:
            method = int(request.args.get('method', '1'))
            f = 'form2_{}'.format(method)
        else:
            data = {'result': 'processing'}
            return jsonify(data)

        try:
            data = {'result': data[module][f]['status']}
            return jsonify(data)

        except Exception as e:
            data = {'result': 'processing'}
            return jsonify(data)
    elif module.upper() == 'CA':
        f = 'form{}'.format(step-1)
        try:
            data = {'result': data[module][f]['status']}
            return jsonify(data)
        except Exception as e:
            data = {'result': 'processing'}
            return jsonify(data)

    data = {'result': 'processing'}
    return jsonify(data)

def calculate_file_md5(file_path):
    """计算文件的 MD5 哈希值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # 以二进制方式读取文件并更新哈希值
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def check_if_file_exists(md5_hash):
    folder_files = os.listdir(UPLOAD_FOLDER)
    for existing_file in folder_files:
        existing_file_name, _ = os.path.splitext(existing_file)
        if existing_file_name == md5_hash:
            print("File already exists:", existing_file)
            return True, os.path.join(UPLOAD_FOLDER, existing_file)
    return False, ''


def save_files(files, save_path, version):
    """
    :param files:
    :return: files info path

    list of file info: {
        'version': version,
        'filename': filename, # 用户上传的原始文件名
        'md5': md5,           # 将文件名
    }
    """

    info = []

    # 先存到临时文件，再计算每一个 md5 值。以此判断避免重复占用空间。
    for file in files:
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            to_file_tmp = os.path.join(TMP_FOLDER, filename)
            file.save(to_file_tmp)
            print("uploaded to " + to_file_tmp)

            md5_hash = calculate_file_md5(to_file_tmp)

            existed, existed_filename =  check_if_file_exists(md5_hash)
            if not existed:
                extension = filename.rsplit('.', 1)[1].lower()
                existed_filename = os.path.join(UPLOAD_FOLDER, "{}.{}".format(md5_hash, extension))
                file.save(existed_filename)
                shutil.move(to_file_tmp, existed_filename)
                print("saved to " + existed_filename)

            info.append({
                'version': version,
                'filename': filename,
                'md5': md5_hash,
                'existed_filename': existed_filename,
            })
    to_file_info = os.path.join(save_path, "fileinfo.json")
    with open(to_file_info, 'w') as f:
        f.write(json.dumps(info))
    return to_file_info


"""
这里是所有表单提交的入口
"""
@app.route('/upload', methods=['POST'])
def upload_file():
    version = None

    try:
        print(request.form)
        module = request.form.get('module')

        print("module: " + module)
    except Exception as e:
        print(str(e))
        module = "None"

    if module.upper() == 'CV':
        step = request.form.get('step', '0')
        func = int(request.form.get('func', '0'))
        if func > 0:
            if func == 4 or func == 5:
                version = request.form.get('version')
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], version)
                files_info = os.path.join(save_path, "fileinfo.json".format(version))
                all_params = request.form.to_dict()
                all_params['files_info'] = files_info

                user_input = {
                    'version': version,
                    'module': module,
                    'step': step,
                    'func': func,
                    'data': all_params
                }
                # 创建子线程，并启动后台任务
                background_thread = threading.Thread(target=background_task, args=(user_input,))
                background_thread.start()

                return {
                    'status': True,
                    'message': 'Success, please wait.',
                    'version': version
                }
            else:
                return jsonify({
                    'status': False,
                    'message': 'One or more files are not allowed.'
                })
        else:
            if step == '1':
                version = "version_" + datetime.now().strftime("%m%d_%H%M%S")
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], version)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                data_path = os.path.join('outputs', version)
                if not os.path.exists(data_path):
                    os.makedirs(data_path, exist_ok=True)
                # data_file = os.path.join(data_path, 'data.json')
                # if not os.path.exists(data_file):
                #     data = CV.demo_data()
                #     data['version'] = version
                #
                #     with open(data_file, 'w') as f:
                #         json.dump(data, f)

                if 'files[]' not in request.files:
                    return 'No file part'

                files = request.files.getlist('files[]')
                files_info = save_files(files, save_path, version)
                all_params = request.form.to_dict()
                all_params['files_info'] = files_info

                # sigma = request.form.get('sigma')
                # print("sigma: " + str(sigma))

                user_input = {
                    'version': version,
                    'module': module,
                    'step': step,
                    'data': all_params
                }
                # 创建子线程，并启动后台任务
                background_thread = threading.Thread(target=background_task, args=(user_input,))
                background_thread.start()

                return jsonify({
                    'status': True,
                    'message': 'Success, please wait.',
                    'version': version
                })
            else:
                version = request.form.get('version')
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], version)
                files_info = os.path.join(save_path, "fileinfo.json".format(version))

                all_params = request.form.to_dict()
                all_params['files_info'] = files_info

                user_input = {
                    'version': version,
                    'module': module,
                    'step': step,
                    'data': all_params
                }
                # 创建子线程，并启动后台任务
                background_thread = threading.Thread(target=background_task, args=(user_input,))
                background_thread.start()

                return {
                    'status': True,
                    'message': 'Success, please wait.',
                    'version': version
                }
            # else:
            #     return jsonify({
            #         'status': False,
            #         'message': 'One or more files are not allowed.'
            #     })
    elif module.upper() == 'HDV':
        step = request.form.get('step', '1')
        if step == '1':
            version = "version_" + datetime.now().strftime("%m%d_%H%M%S")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], version)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            if 'files[]' not in request.files:
                return 'No file part'

            files = request.files.getlist('files[]')
            files_info = save_files(files, save_path, version)

            sigma = float(request.form.get('sigma', 10))

            user_input = {
                'version': version,
                'module': module,
                'step': step,
                'data': {
                    'files_info': files_info,
                    'sigma': sigma
                }
            }
            # 创建子线程，并启动后台任务
            background_thread = threading.Thread(target=background_task, args=(user_input,))
            background_thread.start()

            return jsonify({
                'status': True,
                'message': 'Success, please wait.',
                'version': version
            })
        elif step == '2':
            version = request.form.get('version')
            all_params = request.form.to_dict()
            user_input = {
                'version': version,
                'module': module,
                'step': step,
                'data': all_params
            }
            # 创建子线程，并启动后台任务
            background_thread = threading.Thread(target=background_task, args=(user_input,))
            background_thread.start()

    elif module.upper() == 'CA':
        step = request.form.get('step', '1')
        if step == '1':
            version = "version_" + datetime.now().strftime("%m%d_%H%M%S")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], version)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            data_path = os.path.join('outputs', version)
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)
            # data_file = os.path.join(data_path, 'data.json')
            # if not os.path.exists(data_file):
            #     data = CV.demo_data()
            #     data['version'] = version
            #
            #     with open(data_file, 'w') as f:
            #         json.dump(data, f)

            if 'files[]' not in request.files:
                return 'No file part'

            files = request.files.getlist('files[]')
            files_info = save_files(files, save_path, version)

            # sigma = request.form.get('sigma')
            # print("sigma: " + str(sigma))

            user_input = {
                'version': version,
                'module': module,
                'step': step,
                'data': {
                    'files_info': files_info,
                    # 'sigma': float(sigma)
                }
            }
            # 创建子线程，并启动后台任务
            background_thread = threading.Thread(target=background_task, args=(user_input,))
            background_thread.start()

            return jsonify({
                'status': True,
                'message': 'Success, please wait.',
                'version': version
            })
        elif step == '2':
            version = request.form.get('version')
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], version)
            files_info = os.path.join(save_path, "fileinfo_{}.json".format(version))

            user_input = {
                'version': version,
                'module': module,
                'step': step,
                'data': {
                    'files_info': files_info,
                    'interval': 5,
                    'n': int(request.form.get('input_n', 1)),
                    'a': float(request.form.get('input_a', 0.07068583470577035)),
                    'c': float(request.form.get('input_c', 0.000966e-3)),
                    'x_range': request.form.get('input_range', ''),
                }
            }
            # 创建子线程，并启动后台任务
            background_thread = threading.Thread(target=background_task, args=(user_input,))
            background_thread.start()

            return {
                'status': True,
                'message': 'Success, please wait.',
                'version': version
            }
        else:
            return jsonify({
                'status': False,
                'message': 'One or more files are not allowed.'
            })

    return jsonify({
        'status': True,
        'message': 'Success, please wait.',
        'version': version
    })


@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, '../outputs'), filename)

@app.route('/outputs/<version>/<filename>')
def uploaded_file2(version, filename):
    return send_from_directory(os.path.join(BASE_DIR, '../outputs/{}'.format(version)), filename)

@app.route('/files/<filename>')
def files(filename):
    # return send_from_directory('data/example_files', filename)
    return send_file(os.path.join( BASE_DIR, '../data/example_files/{}'.format(filename) )  , as_attachment=True)


def background_task(param):
    print("Background task started with parameter:", param)

    if param['module'].upper() == 'CV':
        if 'func' in param.keys() and param['func'] > 0:
            if param['func'] == 4:
                all_params = param['data']
                c = CV(version=all_params['version'], files_info=all_params['files_info'])
                c.start4(all_params)
            elif param['func'] == 5:
                all_params = param['data']
                c = CV(version=all_params['version'], files_info=all_params['files_info'])
                c.start5(all_params)
        else:
            if param['step'] == '1':
                all_params = param['data']
                c = CV(version=param['version'], files_info=all_params['files_info'])
                c.start1(all_params)
            elif param['step'] == '2':
                all_params = param['data']
                c = CV(version=param['version'], files_info=all_params['files_info'])
                c.start2(all_params)
            elif param['step'] == '3':
                all_params = param['data']
                c = CV(version=param['version'], files_info=all_params['files_info'])
                c.start3(all_params)
    elif param['module'].upper() == 'HDV':
        if param['step'] == '1':
            d = param['data']
            print("=======")
            print(d)
            h = HDV(version=param['version'])
            h.step1(sigma=d['sigma'])
        elif param['step'] == '2':
            d = param['data']
            h = HDV(version=param['version'])
            if d['method'] == '1':
                h.step2_1(d)
            else:
                h.step2_2(d)
    elif param['module'].upper() == 'CA':
        if param['step'] == '1':
            d = param['data']
            h = CA(version=param['version'])
            h.step1()
        elif param['step'] == '2':
            d = param['data']
            h = CA(version=param['version'])
            h.step2(d['interval'], d['n'], d['a'], d['c'], d['x_range'])

    print("Background task completed")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8080)
