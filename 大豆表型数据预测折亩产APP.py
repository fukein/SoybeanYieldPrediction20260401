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

# ===================== 侧边栏菜单 =====================
with st.sidebar:
    st.title("🌱 系统菜单")
    menu = st.radio("", ["产量预测", "关于系统"])

# ===================== 加载模型 =====================
try:
    model = joblib.load('大豆表型数据best_model_XGB.pkl')
except:
    st.error("模型文件未找到，请检查路径")
    st.stop()

# ===================== 特征中英文独立映射 =====================
feature_list = [
    "株高", "结荚高度", "主茎节数", "有效分枝", "有效荚数", "无效荚数",
    "单株生物产量", "单株粒数", "单株粒重", "百粒重计算", "每荚粒数",
    "经济系数", "粒形", "脐色", "籽粒光泽", "生育天数", "水分", "蛋白", "脂肪", "抗倒"
]

# 极简英文（SHAP图专用）
en_names = [
    "PH","PHgt","Nod","Br","Pod","UPod",
    "Bio","SN","SW","100W","SPod",
    "HI","Sh","Hc","Gls","Day","Mo","Pro","Fat","Lod"
]

# 表格显示（英文+中文）
table_names = [
    "PH(株高)","PHgt(结荚高度)","Nod(主茎节数)","Br(有效分枝)","Pod(有效荚数)","UPod(无效荚数)",
    "Bio(单株生物产量)","SN(单株粒数)","SW(单株粒重)","100W(百粒重)","SPod(每荚粒数)",
    "HI(经济系数)","Sh(粒形)","Hc(脐色)","Gls(光泽)","Day(生育天)","Mo(水分)","Pro(蛋白)","Fat(脂肪)","Lod(抗倒)"
]

# 合理默认初始值（行业常用均值，无上限）
default_values = [
    95.0, 22.0, 16.0, 2.0, 42.0, 5.0,
    75.0, 90.0, 15.0, 18.0, 2.3,
    0.38, 2.0, 2.0, 2.0, 115.0, 12.5, 41.0, 20.0, 3.0
]

# ===================== 菜单1：产量预测 =====================
if menu == "产量预测":
    st.title("🌱 大豆表型数据产量预测系统")

    # 输入面板（无上限，仅≥0，带合理默认值）
    st.markdown("<div class='card'><div class='section-title'>表型特征输入</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    input_values = {}

    for i, feat in enumerate(feature_list):
        with cols[i % 4]:
            input_values[feat] = st.number_input(
                feat,
                min_value=0.0,
                value=default_values[i],
                step=0.1
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # 预测按钮
    _, c, _ = st.columns([1, 1, 1])
    with c:
        run = st.button("🔍 预测产量", use_container_width=True)

    # 预测计算
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
                "en_feats": en_names,
                "table_feats": table_names
            }

    # 展示结果
    if "data" in st.session_state:
        d = st.session_state.data

        # 预测结果
        st.markdown(f"""
        <div class="result-card">
            预测折亩产：{d['yield']} kg/亩
        </div>
        """, unsafe_allow_html=True)

        # ========== 第一行：整行显示 SHAP 力图（超扁，宽高比≈6:1～8:1）==========
        st.markdown("<div class='card'><div class='section-title'>SHAP 力图解释</div>", unsafe_allow_html=True)
        shap.force_plot(
            base_value=d['base'],
            shap_values=d['shap'],
            features=d['input'].iloc[0].values,
            feature_names=d['en_feats'],
            matplotlib=True,
            figsize=(24, 3.5)  # ✅ 宽24，高3.5 → 宽高比≈6.8:1（非常扁）
        )
        st.pyplot(plt.gcf())
        st.caption("红色：正向增产 ｜ 蓝色：负向减产")
        st.markdown("</div>", unsafe_allow_html=True)

        # ========== 第二行：整行显示 输入特征 ==========
        st.markdown("<div class='card'><div class='section-title'>输入特征</div>", unsafe_allow_html=True)
        show_df = pd.DataFrame({
            "特征": d['table_feats'],
            "数值": d['input'].iloc[0].values
        })
        st.dataframe(show_df, use_container_width=True, height=500)
        st.markdown("</div>", unsafe_allow_html=True)

        # ========== 第三行：特征贡献度 ==========
        st.markdown("<div class='card'><div class='section-title'>特征贡献度排序</div>", unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            "特征": d['table_feats'],
            "贡献值": d['shap']
        })
        imp_df["绝对影响"] = np.abs(imp_df["贡献值"])
        imp_df = imp_df.sort_values("绝对影响", ascending=False).drop(columns="绝对影响")
        st.dataframe(imp_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===================== 关于系统 =====================
elif menu == "关于系统":
    st.title("ℹ️ 系统说明")
    st.markdown("### 基于AutoML的大豆产量预测模型")
    st.markdown("- 模型：XGBoost")
    st.markdown("- SHAP力图解释特征边际贡献")
    st.markdown("- 用途：大豆表型性状→折亩产预测、育种辅助决策")
