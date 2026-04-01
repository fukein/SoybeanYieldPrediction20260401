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

# ===================== 特征：中文 + 英文映射（画图/表格用英文+中文备注） =====================
feature_cn_en = {
    "株高": "PlantH (株高)",
    "结荚高度": "PodH (结荚高度)",
    "主茎节数": "Nodes (主茎节数)",
    "有效分枝": "Brans (有效分枝)",
    "有效荚数": "PodNum (有效荚数)",
    "无效荚数": "Unpod (无效荚数)",
    "单株生物产量": "BioYld (单株生物产量)",
    "单株粒数": "SeedNum (单株粒数)",
    "单株粒重": "SeedWt (单株粒重)",
    "百粒重计算": "100Wt (百粒重)",
    "每荚粒数": "SeedPod (每荚粒数)",
    "经济系数": "HarvIdx (经济系数)",
    "粒形": "Shape (粒形)",
    "脐色": "HilumC (脐色)",
    "籽粒光泽": "SeedGl (籽粒光泽)",
    "生育天数": "GrowDay (生育天数)",
    "水分": "Moist (水分)",
    "蛋白": "Protein (蛋白)",
    "脂肪": "Oil (脂肪)",
    "抗倒": "Lodg (抗倒)"
}

feature_list = list(feature_cn_en.keys())
en_feature_list = list(feature_cn_en.values())

feature_ranges = {
    "株高": (50.0, 150.0, 95.0), "结荚高度": (10.0, 40.0, 22.0),
    "主茎节数": (8.0, 25.0, 16.0), "有效分枝": (0.0, 5.0, 2.0),
    "有效荚数": (10.0, 80.0, 42.0), "无效荚数": (0.0, 20.0, 5.0),
    "单株生物产量": (20.0, 150.0, 75.0), "单株粒数": (20.0, 200.0, 90.0),
    "单株粒重": (5.0, 30.0, 15.0), "百粒重计算": (10.0, 30.0, 18.0),
    "每荚粒数": (1.5, 3.5, 2.3), "经济系数": (0.2, 0.6, 0.38),
    "粒形": (1.0, 3.0, 2.0), "脐色": (1.0, 4.0, 2.0), "籽粒光泽": (1.0, 3.0, 2.0),
    "生育天数": (90.0, 130.0, 115.0), "水分": (8.0, 15.0, 12.5),
    "蛋白": (35.0, 50.0, 41.0), "脂肪": (15.0, 24.0, 20.0),
    "抗倒": (1.0, 5.0, 3.0)
}

# ===================== 菜单1：产量预测 =====================
if menu == "产量预测":
    st.title("🌱 大豆表型数据产量预测系统")

    # 输入面板（保留中文）
    st.markdown("<div class='card'><div class='section-title'>表型特征输入</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    input_values = {}

    for i, feat in enumerate(feature_list):
        with cols[i % 4]:
            min_v, max_v, def_v = feature_ranges[feat]
            input_values[feat] = st.number_input(feat, min_value=min_v, max_value=max_v, value=def_v, step=0.1)
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
            st.markdown("<div class='card'><div class='section-title'>输入特征（英文+中文）</div>", unsafe_allow_html=True)
            show_df = d['input'].T.reset_index()
            show_df.columns = ['特征', '数值']
            show_df['特征'] = show_df['特征'].map(feature_cn_en)  # 替换为英文+中文
            st.dataframe(show_df, use_container_width=True, height=500)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'><div class='section-title'>SHAP 力图解释</div>", unsafe_allow_html=True)
            # SHAP绘图使用英文名称
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

        # 特征重要性（英文+中文）
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
    st.markdown("- 解释方法：SHAP力图（英文显示，括号标注中文）")
    st.markdown("- 用途：大豆表型性状→折亩产预测、育种辅助决策")
