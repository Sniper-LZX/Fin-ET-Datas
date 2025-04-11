from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import json
import pickle
import re
from Tool.PDF2TXT import convert_pdf_to_txt  # 导入转换函数
from Tool.Texts_Classify import sentence_trans  # 导入段落划分函数
from Tool.ZhipuControlAPI import llm_extract  # 导入大模型信息抽取函数
from Tool.Draw_Pictures import plot_directed_graph

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# 确保上传目录存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 将 os 模块的功能传递给模板
@app.context_processor
def inject_os():
    return dict(os=os)


@app.route('/')
def index():
    return render_template('index.html')


# 事件提取专用页面
@app.route('/event_extract')
def event_extract():
    return render_template('extract.html')  # 新的事件提取页面


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('未选择文件！', 'error')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('未选择文件！', 'error')
        return redirect(url_for('index'))

    # 检查文件是否已经存在
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    if os.path.exists(file_path):
        flash('已经上传！', 'error')
    else:
        # 保存文件
        file.save(file_path)
        flash('上传文件成功！', 'success')

    return redirect(url_for('index'))


# 异步处理 PDF 转换
def async_convert_pdf_to_txt(pdf_file_path, txt_file_path):
    try:
        success, message = convert_pdf_to_txt(pdf_file_path, txt_file_path)
        print(f"转换结果: {message}")
    except Exception as e:
        print(f"转换失败: {e}")


@app.route('/convert', methods=['POST'])
def convert_pdf_to_txt_route():
    # 获取上传的 PDF 文件
    pdf_filename = request.form.get('pdf_filename')
    if not pdf_filename:
        return jsonify({"success": False, "message": "未选择 PDF 文件！"})

    # 构建文件路径
    pdf_file_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
    txt_file_path = os.path.join(UPLOAD_FOLDER, pdf_filename.replace('.pdf', '.txt'))

    # 如果目标文件已存在，直接返回
    if os.path.exists(txt_file_path):
        return jsonify({
            "success": True,
            "message": "已经转换！",
            "txt_file_path": f"uploads/{os.path.basename(txt_file_path)}"  # 返回以 uploads/ 开头的路径
        })

    # 调用 PDF2TXT.py 中的转换函数
    success, message = convert_pdf_to_txt(pdf_file_path, txt_file_path)
    if success:
        return jsonify({
            "success": True,
            "message": "转换完成！",
            "txt_file_path": f"uploads/{os.path.basename(txt_file_path)}"  # 返回以 uploads/ 开头的路径
        })
    else:
        return jsonify({"success": False, "message": message})


# 异步处理段落划分
def async_classify_text(txt_file_path, similar_file_path):
    try:
        sentence_trans(txt_file_path, similar_file_path)
        print("段落划分完成！")
    except Exception as e:
        print(f"段落划分失败: {e}")


@app.route('/classify', methods=['POST'])
def classify_text():
    # 获取上传的 TXT 文件
    txt_filename = request.form.get('txt_filename')
    if not txt_filename:
        return jsonify({"success": False, "message": "未选择 TXT 文件！"})

    # 构建文件路径
    txt_file_path = os.path.join(UPLOAD_FOLDER, txt_filename)
    similar_file_path = os.path.join(UPLOAD_FOLDER, txt_filename.replace('.txt', '_划分.txt'))

    # 如果目标文件已存在，直接返回
    if os.path.exists(similar_file_path):
        return jsonify({"success": True, "message": "已经划分！"})

    # 调用 texts_classify.py 中的划分函数
    sentence_trans(txt_file_path, similar_file_path)
    return jsonify({"success": True, "message": "划分完成！"})


# 异步处理信息抽取
def async_extract_info(api_key, similar_file_path, json_file_path):
    try:
        llm_extract(api_key, similar_file_path, json_file_path)
        print("大模型抽取完成！")
    except Exception as e:
        print(f"大模型抽取失败: {e}")


@app.route('/extract', methods=['POST'])
def extract_info():
    # 获取用户输入的 API Key
    api_key = request.form.get('api_key')
    if not api_key:
        return jsonify({"success": False, "message": "未输入 API Key！"})

    # 获取上传的 TXT 文件
    txt_filename = request.form.get('txt_filename')
    if not txt_filename:
        return jsonify({"success": False, "message": "未选择 TXT 文件！"})

    # 构建文件路径
    similar_file_path = os.path.join(UPLOAD_FOLDER, txt_filename.replace('.txt', '_划分.txt'))
    json_file_path = os.path.join(UPLOAD_FOLDER, txt_filename.replace('.txt', '.json'))

    # 检查是否已经存在划分后的文件
    if not os.path.exists(similar_file_path):
        return jsonify({"success": False, "message": "请先进行段落划分！"})

    # 调用 ZhipuControlAPI.py 中的抽取函数
    llm_extract(api_key, similar_file_path, json_file_path)
    return jsonify({"success": True, "message": "大模型抽取完成！", "json_file_path": json_file_path.replace('\\', '/')})


@app.route('/file-list')
def file_list():
    # 获取 uploads 文件夹中的文件列表
    files = os.listdir('uploads')
    # 渲染文件列表的 HTML 片段
    return render_template('file_list.html', files=files)


@app.route('/file-content')
def file_content():
    # 获取文件路径（包含 uploads/ 前缀）
    file_path = request.args.get('path')
    if not file_path:
        return "未选择文件。", 400

    # 构建完整路径
    full_path = os.path.join(UPLOAD_FOLDER, file_path.replace('uploads/', '', 1))

    # 标准化路径（解决 Windows 反斜杠问题）
    full_path = os.path.normpath(full_path)

    # 检查文件是否存在
    if not os.path.exists(full_path):
        return f"文件不存在：{full_path}", 404

    # 读取文件内容
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"读取文件失败：{str(e)}", 500

# 加载数据文件
with open('最终数据.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 修改路径为 /fin_et_operations
@app.route('/fin_et_operations')
def fin_et_operations():
    # 获取所有不重复的Report值
    reports = sorted({item['Report'] for item in data})
    return render_template('fin_et_operations.html', reports=reports)

# 获取ID的接口路径保持不变
@app.route('/get_ids')
def get_ids():
    # 获取前端传递的Report参数
    selected_reports = request.args.getlist('reports[]')

    # 筛选对应的Id列表
    ids = set()
    for item in data:
        if item['Report'] in selected_reports:
            ids.add(item['Id'])
    return jsonify(sorted(ids))

# 获取数据的接口路径保持不变
@app.route('/get_data')
def get_data():
    # 获取筛选参数
    selected_reports = request.args.getlist('reports[]')
    selected_ids = request.args.getlist('ids[]')

    # 打印调试信息
    print("Selected Reports:", selected_reports[0])
    print("Selected IDs:", selected_ids[0])

    # 执行数据筛选
    filtered_data = [
        item for item in data
        if item['Report'] == selected_reports[0]
        and item['Id'] == int(selected_ids[0])
    ]

    return jsonify(filtered_data)

# 事件关联图页面路由
@app.route('/event_relation_graph')
def event_relation_graph():
    return render_template('event_relation_graph.html')


# 加载 pkl 文件
with open('235权重.pkl', 'rb') as f:
    pkl_data = pickle.load(f)


# 提供一个 API 接口，返回所有数据
@app.route('/api/data', methods=['GET'])
def get_data_pkl():
    return jsonify(pkl_data)


# 提供一个 API 接口，根据编号返回特定数据
@app.route('/api/data/<int:ith>', methods=['GET'])
def get_data_by_ith(ith):
    for item in pkl_data:
        if item['ith'] == ith:
            return jsonify(item)
    return jsonify({"error": "未找到对应编号的数据"}), 404


# 提供一个 API 接口，随机返回一个数据
@app.route('/api/data/random', methods=['GET'])
def get_random_data():
    import random
    random_item = random.choice(pkl_data)
    return jsonify(random_item)


# 定义 get_text_2_id 函数
def get_text_2_id(ith):
    total_text = []
    for d in pkl_data:
        if d['ith'] == ith:
            all_text = d['context'] + d['candidates']
            for text in all_text:
                for item in data:
                    input = re.sub(r'[^\w\s,，.。:：！？;；（）\n]', '', item["Input"])[:len(text)]
                    if input == text:
                        total_text.append(item["Input"])
                        break
    return total_text


# 提供获取完整文本的接口
@app.route('/get_full_text', methods=['POST'])
def get_full_text():
    ith = request.json.get('ith')
    full_text = get_text_2_id(ith)
    return jsonify(full_text)


# 设置图像保存路径
IMAGE_FOLDER = os.path.join('static', 'images')
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)


@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    # 获取前端传递的矩阵数据
    data = request.json
    graphs = data.get('graphs')  # 获取包含多个矩阵的列表

    if not graphs:
        return jsonify({"error": "未提供矩阵数据"}), 400

    # 生成多张图片并保存
    image_urls = []
    for i, graph in enumerate(graphs):
        matrix = graph[0]  # 每个 graph 是一个列表，其中唯一元素是矩阵
        image_path = os.path.join(IMAGE_FOLDER, f'graph_{i}.png')
        print(image_path)
        plot_directed_graph(matrix, image_path)
        image_urls.append(url_for('static', filename=f'images/graph_{i}.png'))

    # 返回图片路径列表
    return jsonify({"image_urls": image_urls})


if __name__ == '__main__':
    app.run(debug=True)