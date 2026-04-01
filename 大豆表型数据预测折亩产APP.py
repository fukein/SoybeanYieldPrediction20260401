import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# ===================== 页面配置 =====================
st.set_page_config(
    page_title="大豆产量预测系统",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="auto"
)

plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

# ===================== 简洁CSS =====================
st.markdown("""
<style>
.main {
    background-color: #f9f9f9;
}
.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    padding: 20px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    border-left: 4px solid #4472c4;
    padding-left: 10px;
    margin-bottom: 15px;
}
.result-card {
    background: #4472c4;
    color: white;
    padding: 22px;
    border-radius: 8px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin: 15px 0;
}
.stButton>button {
    background-color: #4472c4 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ===================== 侧边栏菜单（中文） =====================
with st.sidebar:
    st.title("🌱 系统菜单")
    menu = st.radio(
        "",
        ["产量预测", "关于系统"]
    )

# ===================== 加载模型 =====================
try:
    model = joblib.load('大豆表型数据best_model_XGB.pkl')
except:
    st.error("模型文件未找到，请检查路径")
    st.stop()

# ===================== 特征：纯英文名称（SHAP专用） =====================
feature_cn_en = {
    "株高": "PlantH",
    "结荚高度": "PodH",
    "主茎节数": "Nodes",
    "有效分枝": "Brans",
    "有效荚数": "PodNum",
    "无效荚数": "Unpod",
    "单株生物产量": "BioYld",
    "单株粒数": "SeedNum",
    "单株粒重": "SeedWt",
    "百粒重计算": "100Wt",
    "每荚粒数": "SeedPod",
    "经济系数": "HarvIdx",
    "粒形": "Shape",
    "脐色": "HilumC",
    "籽粒光泽": "SeedGl",
    "生育天数": "GrowDay",
    "水分": "Moist",
    "蛋白": "Protein",
    "脂肪": "Oil",
    "抗倒": "Lodg"
}

feature_list = list(feature_cn_en.keys())
en_feature_list = list(feature_cn_en.values())

# ===================== 菜单1：产量预测 =====================
if menu == "产量预测":
    st.title("🌱 大豆表型数据产量预测系统")

    # 输入面板：无范围限制，仅要求 > 0
    st.markdown("<div class='card'><div class='section-title'>表型特征输入</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    input_values = {}

    for i, feat in enumerate(feature_list):
        with cols[i % 4]:
            # 无最大值限制，最小值=0，默认值保留原合理值
            input_values[feat] = st.number_input(
                feat, 
                min_value=0.0, 
                value=10.0 if feat in ["粒形","脐色","籽粒光泽","抗倒"] else 50.0, 
                step=0.1
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # 预测按钮
    _, c, _ = st.columns([1, 1, 1])
    with c:
        run = st.button("🔍 预测产量", use_container_width=True)

    # 执行预测
    if run:
        with st.spinner("正在预测..."):
            input_df = pd.DataFrame([input_values])
            pred = model.predict(input_df)[0]

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_df)
            ev = explainer.expected_value

            st.session_state.data = {
                "yield": round(pred, 2),
                "input": input_df,
                "shap": sv[0],
                "base": ev,
                "feats": feature_list,
                "en_feats": en_feature_list
            }

    # 展示结果
    if "data" in st.session_state:
        d = st.session_state.data

        # 产量结果
        st.markdown(f"""
        <div class="result-card">
            预测折亩产：{d['yield']}  kg/亩
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='card'><div class='section-title'>输入特征</div>", unsafe_allow_html=True)
            show_df = d['input'].T.reset_index()
            show_df.columns = ['特征', '数值']
            st.dataframe(show_df, use_container_width=True, height=500)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'><div class='section-title'>SHAP 力图解释</div>", unsafe_allow_html=True)
            # 纯英文SHAP力图，无任何中文
            shap.force_plot(
                base_value=d['base'],
                shap_values=d['shap'],
                features=d['input'].iloc[0].values,
                feature_names=d['en_feats'],
                matplotlib=True,
                figsize=(12, 6)
            )
            st.pyplot(plt.gcf())
            st.caption("红色：正向增产 ｜ 蓝色：负向减产")
            st.markdown("</div>", unsafe_allow_html=True)

        # 特征重要性（纯英文）
        st.markdown("<div class='card'><div class='section-title'>特征贡献度排序</div>", unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            "特征": d['en_feats'],
            "贡献值": d['shap'],
            "绝对影响": np.abs(d['shap'])
        }).sort_values("绝对影响", ascending=False).drop(columns="绝对影响")
        st.dataframe(imp_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== 菜单2：关于系统 =====================
elif menu == "关于系统":
    st.title("ℹ️ 系统说明")
    st.markdown("### 基于AutoML的大豆产量预测模型")
    st.markdown("- 模型：XGBoost")
    st.markdown("- 解释方法：SHAP力图（纯英文）")
    st.markdown("- 用途：大豆表型性状→折亩产预测、育种辅助决策")
