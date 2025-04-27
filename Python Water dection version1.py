import os
import uuid
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def process_data(filepath):
    try:
        # 读取数据
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            df = pd.read_csv(filepath)

        # 数据清洗和类型转换
        required_columns = ['经度', '纬度', 'DO', 'Turbidity', '温度', 'PH值', 'COD', 'NH3-N']
        df = df[required_columns].dropna()
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # 特征工程
        features = df[['DO', 'Turbidity', '温度', 'PH值', 'COD', 'NH3-N']]

        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        # 聚类分析（强制转换为整数）
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['cluster'] = kmeans.fit_predict(X).astype(int)  # 关键修复点

        # 计算聚类中心
        cluster_centers = df.groupby('cluster')[['经度', '纬度']].mean().reset_index()

        # 构建返回数据（显式类型转换）
        center_lng = float(df['经度'].mean())
        center_lat = float(df['纬度'].mean())

        points = []
        for _, row in df.iterrows():
            points.append({
                "lng": float(row['经度']),
                "lat": float(row['纬度']),
                "color": ['#ff0000', '#0000ff', '#00ff00'][int(row['cluster'])]  # 强制转换索引
            })

        sources = []
        for _, center in cluster_centers.iterrows():
            sources.append({
                "lng": float(center['经度']),
                "lat": float(center['纬度']),
                "cluster": int(center['cluster'])  # 确保为整数
            })

        return {
            "center": [center_lng, center_lat],
            "points": points,
            "sources": sources
        }

    except Exception as e:
        raise RuntimeError(f"数据处理失败: {str(e)}")


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未上传文件'}), 400

        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': '无效文件'}), 400

        # 生成唯一文件名
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 处理数据
        result = process_data(filepath)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)