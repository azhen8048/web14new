import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xgboost
from xgboost import XGBClassifier

# 加载保存的随机森林模型
model = joblib.load('xgb_model.pkl')

# 特征范围定义（根据新提供的变量列表）
feature_ranges = {
    "weight": {
        "type": "numerical",
        "min": 0.0,
        "max": 300.0,
        "default": 70.0,
        "unit": "kg"
    },
    "WBC": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "unit": "×10⁹/L"
    },
    "PO2": {
        "type": "numerical",
        "min": 0.0,
        "max": 300.0,
        "default": 90.0,
        "unit": "mmHg"
    },
    "APSIII": {
        "type": "numerical",
        "min": 0.0,
        "max": 120.0,
        "default": 50.0,
        "unit": "None"
    },
    "SBP": {
        "type": "numerical",
        "min": 0.0,
        "max": 300.0,
        "default": 120.0,
        "unit": "mmHg"
    },
    "BUN": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "unit": "mmol/L"
    },
    "Glucose": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 10.0,
        "unit": "mmol/L"
    },
    "SOFA": {
        "type": "numerical",
        "min": 0.0,
        "max": 24.0,
        "default": 5.0,
        "unit": "None"
    },
    "Albumin": {
        "type": "numerical",
        "min": 0.0,
        "max": 100.0,
        "default": 40.0,
        "unit": "g/L"
    },
    "heartrate": {
        "type": "numerical",
        "min": 0.0,
        "max": 200.0,
        "default": 80.0,
        "unit": "bpm"
    },
    "Plateletcount": {
        "type": "numerical",
        "min": 0.0,
        "max": 1000.0,
        "default": 300.0,
        "unit": "×10⁹/L"
    },
    "Norepinephrine": {
        "type": "categorical",
        "options": [0, 1],
        "default": 0
    }
}

# Streamlit 界面
st.header("Enter the following feature values:")
feature_values = []

for feature, properties in feature_ranges.items():
    display_name = feature.replace('weight', 'Weight').replace('heartrate', 'Heartrate')

    if properties["type"] == "numerical":
        unit = properties.get("unit")
        unit_str = f"{unit}, " if unit is not None else ""   # 关键改动
        label = f"{display_name} ({unit_str}{properties['min']}–{properties['max']})"
        value = st.number_input(
            label=label,
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"])
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{display_name} (Select a value)",
            options=properties["options"]
        )
    feature_values.append(value)
    
# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of septic shock is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))

    # 设置Times New Roman斜体加粗
    try:
        prop = font_manager.FontProperties(
            family='Times New Roman',
            style='italic',
            weight='bold',
            size=16
        )
        ax.text(
            0.5, 0.5, text,
            fontproperties=prop,
            ha='center', va='center',
            transform=ax.transAxes
        )
    except:
        ax.text(
            0.5, 0.5, text,
            fontsize=16,
            ha='center', va='center',
            style='italic',
            weight='bold',
            family='serif',
            transform=ax.transAxes
        )

    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300, transparent=True)
    st.image("prediction_text.png")

        # ===== 计算 SHAP 值 =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame([feature_values], columns=feature_ranges.keys())
    )

    # ---- 统一成 2-D ----
    if isinstance(shap_values, list):                 
        shap_values = np.array(shap_values[1])        
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = shap_values[:, :, 0]           

    baseline = float(explainer.expected_value)       
    sv = shap_values[0]                              

    # ---- SHAP 力图 ----
    shap_fig = shap.force_plot(
        baseline,
        sv,
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")