# -*- coding: utf-8 -*-
import folium
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import webbrowser
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 获取源文件所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 生成唯一文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------
# 步骤1：数据预处理与时间解析
# ---------------------------
try:
    # 读取原始数据
    raw_df = pd.read_csv(
        "C:\\Users\\lenovo\\Desktop\\water_quality_data.csv",
        skiprows=2,
        encoding='gbk'
    )

    # 清洗列名和时间数据
    raw_df = raw_df.rename(columns={'日期': '时间'})
    raw_df['时间'] = raw_df['时间'].str.strip()

    # 转换时间格式
    raw_df['时间'] = pd.to_datetime(
        raw_df['时间'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # 清除无效数据
    df = raw_df.dropna(subset=['时间', '纬度', '经度', 'COD'])
    df = df.sort_values('时间')

    if df.empty:
        raise ValueError("清洗后无有效数据，请检查数据完整性")

    # 对经纬度进行四舍五入，保留小数点后4位
    df['纬度'] = df['纬度'].round(5)
    df['经度'] = df['经度'].round(5)

    print(f"有效监测点数量：{len(df.groupby(['纬度', '经度']))}")

except Exception as e:
    print(f"数据加载失败：{str(e)}")
    exit()


# ---------------------------
# 步骤2：污染物趋势预测
# ---------------------------
def predict_trend(dataframe):
    df = dataframe.copy()
    df['预测COD'] = np.nan
    df['趋势'] = ''

    valid_groups = 0
    for _, group in df.groupby(['纬度', '经度']):
        if len(group) < 3:
            continue

        # 时间序列处理
        time_diff = (group['时间'] - group['时间'].min()).dt.days
        X = time_diff.values.reshape(-1, 1)
        y = group['COD'].values

        # 线性回归预测
        model = LinearRegression()
        model.fit(X, y)
        future_day = X[-1] + 30
        predicted = model.predict([[future_day[0]]])[0]

        # 标注趋势
        trend = "↑" if predicted > y[-1] else "↓"
        latest_idx = group.index[-1]
        df.at[latest_idx, '预测COD'] = predicted
        df.at[latest_idx, '趋势'] = trend
        valid_groups += 1

    if valid_groups == 0:
        raise ValueError("每个监测点至少需要3条时序数据")

    return df.dropna(subset=['预测COD'])


try:
    df = predict_trend(df)
    print(f"可预测监测点数量：{len(df)}")
except Exception as e:
    print(f"趋势预测失败：{str(e)}")
    exit()

# ---------------------------
# 步骤3：数据标准化与聚类分析
# ---------------------------
features = df[['COD', '氨氮', '溶解氧(mg/L)', 'PH值', '预测COD']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 肘部法则确定最佳k值
distortions = []
K_range = range(1, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, distortions, 'bx-')
plt.xlabel('簇数量 (k)')
plt.ylabel('惯性值')
plt.title('肘部法则 - 选择最优聚类数量')
plt.show()

# 用户输入验证
while True:
    try:
        k = int(input("请输入最佳聚类数量 (1-7): "))
        if 1 <= k <= 7:
            break
        print("请输入1-7之间的整数")
    except ValueError:
        print("无效输入")

# 执行聚类
kmeans = KMeans(n_clusters=k, random_state=42)
df['污染等级'] = kmeans.fit_predict(scaled_features)


# ---------------------------
# 步骤4：动态地图生成
# ---------------------------
def generate_map(dataframe):
    # 创建底图
    map_center = [dataframe['纬度'].mean(), dataframe['经度'].mean()]
    m = folium.Map(location=map_center,
                   zoom_start=14,
                   tiles='CartoDB positron')

    # 颜色映射
    colors = ['#00FF00', '#FFFF00', '#FFA500',
              '#FF0000', '#8B0000', '#800000', '#000000']

    # 添加监测点标记
    for _, row in dataframe.iterrows():
        # 污染等级圆圈
        folium.Circle(
            location=[row['纬度'], row['经度']],
            radius=80,
            color=colors[row['污染等级'] % 7],
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

        # 趋势箭头
        icon = folium.DivIcon(
            html=f"""
            <div style="
                font-size: 24px;
                color: {colors[row['污染等级'] % 7]};
                transform: rotate({-45 if row['趋势'] == '↑' else 45}deg);
            ">▶</div>
            """
        )
        folium.Marker(
            [row['纬度'] + 0.00015, row['经度'] + 0.00015],
            icon=icon
        ).add_to(m)

    # 保存文件到源文件所在的目录
    filename = os.path.join(script_dir, f'水质报告_{timestamp}.html')
    m.save(filename)
    return filename


try:
    output_file = generate_map(df)
    print(f"报告已生成：{output_file}")
    webbrowser.open(output_file)
except Exception as e:
    print(f"地图生成失败：{str(e)}")
