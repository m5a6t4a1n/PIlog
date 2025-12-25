import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="PI预测模型",
    page_icon="🏥",
    layout="wide"
)

# 作者和单位信息
AUTHOR_INFO = {
    "author": "石层层",
    "institution": "山东药品食品职业学院"
}

# 加载保存的Logistic Regression模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('log_reg.pkl')
        return model
    except FileNotFoundError:
        st.error("模型文件 'log_reg.pkl' 未找到。请确保模型文件已上传。")
        return None

model = load_model()

# 特征缩写映射
feature_abbreviations = {
    "FCTI": "FCTI",
    "Age": "Age",
    "Ser": "Ser",
    "Fra": "Fra",
    "Air": "Air",
    "Com": "Com",
    "PCAT": "PCAT",
    "Mlu": "Mlu"
}

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
   "FCTI": {"type": "numerical", "min": 0, "max": 40, "default": 21, "label": "FCTI总分"},
    "Age": {"type": "numerical", "min": 70, "max": 99, "default": 78, "label": "年龄（岁）"},
    "Ser": {"type": "numerical", "min": 20, "max": 60, "default": 21, "label": "血清白蛋白"},
    "Fra": {"type": "categorical", "options": [0,1,2,3,4,5,6,7,8,9,10,11,12,13], "default": 9, "label": "骨折类型", 
            "option_labels": {0: "颈椎骨折", 1: "胸椎骨折",2: "腰椎骨折", 
                              3: "股骨颈骨折", 4: "股骨粗隆间骨折", 5: "股骨干骨折", 6: "胫腓骨上段骨折",
                              7: "尾骨粉碎性骨折", 8: "骶髂关节脱位", 9: "髋骨骨折", 
                              10: "髌骨粉碎性骨折", 11: "髋关节内骨折", 12: "脆性骨折", 13: "其他"}},
    "Air": {"type": "categorical", "options": [0, 1], "default": 0, "label": "气垫床/充气床垫", "option_labels": {0: "未使用", 1: "使用"}},
    "Com": {"type": "numerical", "min": 0, "max": 8, "default": 2, "label": "合并症数量"},
    "PCAT": {"type": "numerical", "min": 1, "max": 4, "default": 3, "label": "PCAT总分"},
    "Mlu": {"type": "categorical", "options": [0, 1], "default": 0, "label": "多发性骨折", "option_labels": {0: "否", 1: "是"}},
}

# Streamlit 界面
st.title('"医院-家庭-社区"三区联合延续护理模式下的老年骨折卧床患者PI风险预测模型')

# 添加作者信息（在主标题下方）
st.markdown(f"""
<div style='text-align: center; color: #666; margin-top: -10px; margin-bottom: 20px;'>
    开发单位：{AUTHOR_INFO["institution"]} | 作者：{AUTHOR_INFO["author"]}
</div>
""", unsafe_allow_html=True)

# 添加说明文本
st.markdown("""
本应用基于机器学习模型预测在"医院-家庭-社区"三区联合延续护理模式下的老年骨折卧床患者PI风险。
请在下方的表单中输入患者的临床指标，然后点击"开始预测"按钮。
""")

# 动态生成输入项
st.header("请输入患者临床指标:")
feature_values = []

# 创建两列布局，使界面更紧凑
col1, col2 = st.columns(2)

features_list = list(feature_ranges.keys())
half_point = len(features_list) // 2

for i, feature in enumerate(features_list):
    properties = feature_ranges[feature]
    
    # 根据位置选择列
    if i < half_point:
        with col1:
            if properties["type"] == "numerical":
                value = st.number_input(
                    label=f"{properties['label']}",
                    min_value=float(properties["min"]),
                    max_value=float(properties["max"]),
                    value=float(properties["default"]),
                    help=f"范围: {properties['min']} - {properties['max']}"
                )
            elif properties["type"] == "categorical":
                # 对于分类变量，使用选择框并显示中文标签
                option_labels = properties.get("option_labels", {k: str(k) for k in properties["options"]})
                selected_label = st.selectbox(
                    label=f"{properties['label']}",
                    options=properties["options"],
                    format_func=lambda x: option_labels[x],
                    index=properties["options"].index(properties["default"])
                )
                value = selected_label
            feature_values.append(value)
    else:
        with col2:
            if properties["type"] == "numerical":
                value = st.number_input(
                    label=f"{properties['label']}",
                    min_value=float(properties["min"]),
                    max_value=float(properties["max"]),
                    value=float(properties["default"]),
                    help=f"范围: {properties['min']} - {properties['max']}"
                )
            elif properties["type"] == "categorical":
                option_labels = properties.get("option_labels", {k: str(k) for k in properties["options"]})
                selected_label = st.selectbox(
                    label=f"{properties['label']}",
                    options=properties["options"],
                    format_func=lambda x: option_labels[x],
                    index=properties["options"].index(properties["default"])
                )
                value = selected_label
            feature_values.append(value)

# 添加一个分隔线
st.markdown("---")

# 预测与 SHAP 可视化
if model is not None and st.button("开始预测", type="primary"):
    # 显示加载指示器
    with st.spinner('模型正在计算中，请稍候...'):
        # 转换为模型输入格式
        features = np.array([feature_values])
        
        # 创建DataFrame用于模型预测
        features_df = pd.DataFrame([feature_values], columns=features_list)

        # 模型预测
        predicted_class = model.predict(features_df)[0]
        predicted_proba = model.predict_proba(features_df)[0]

        # 提取预测的类别概率
        probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    st.subheader("预测结果")
    
    # 使用进度条和指标显示概率
    st.metric(label="PI发生概率", value=f"{probability:.2f}%")
    st.progress(int(probability))
    
    # 添加风险等级解读
    if probability < 20:
        risk_level = "低风险"
        color = "green"
    elif probability < 50:
        risk_level = "中风险"
        color = "orange"
    else:
        risk_level = "高风险"
        color = "red"
    
    st.markdown(f"<h4 style='color: {color};'>风险等级: {risk_level}</h4>", unsafe_allow_html=True)
    
    # 预测类别解释
    if predicted_class == 0:
        st.info("预测结果：该患者发生PI的风险较低")
    else:
        st.warning("预测结果：该患者发生PI的风险较高")
    
    # 创建用于SHAP的DataFrame，使用缩写作为列名
    shap_df = pd.DataFrame([feature_values], columns=features_list)
    shap_df.columns = [feature_abbreviations[col] for col in shap_df.columns]
    
    # 计算 SHAP 值
    with st.spinner('正在生成模型解释图...'):
        try:
            # 创建背景数据集（使用训练数据或创建虚拟数据）
            # 这里我们创建一个虚拟背景数据集，包含可能的特征值范围
            background_data = []
            for feature in features_list:
                prop = feature_ranges[feature]
                if prop["type"] == "numerical":
                    # 对于数值特征，使用默认值
                    background_data.append([prop["default"]])
                else:
                    # 对于分类特征，使用默认选项
                    background_data.append([prop["default"]])
            
            # 转换背景数据为DataFrame
            background_df = pd.DataFrame([data[0] for data in background_data], 
                                         index=features_list).T
            background_df.columns = [feature_abbreviations[col] for col in background_df.columns]
            
            # 对于Logistic Regression，使用LinearExplainer
            explainer = shap.LinearExplainer(model, background_df)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(shap_df)
            
            # 确保SHAP值是二维数组
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # 对于二分类逻辑回归，通常取第二个（正类）的SHAP值
                shap_values_array = shap_values[1]
            elif len(shap_values.shape) == 3:
                # 如果是三维数组，取正类的SHAP值
                shap_values_array = shap_values[:, :, 1]
            else:
                shap_values_array = shap_values
            
            # 生成 SHAP 力图
            plt.figure(figsize=(12, 4), dpi=300)
            shap.force_plot(
                explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                shap_values_array[0],
                shap_df.iloc[0].values,
                feature_names=shap_df.columns.tolist(),
                matplotlib=True,
                show=False
            )
            
            # 获取当前图形
            plt.title(f"SHAP力图 - PI预测概率: {probability:.2f}%", fontsize=12, pad=20)
            plt.tight_layout()
            
            # 保存力图为图像
            buf_force = BytesIO()
            plt.savefig(buf_force, format="png", bbox_inches="tight", dpi=300)
            plt.close()
            
            # 生成 SHAP 瀑布图
            plt.figure(figsize=(12, 6), dpi=300)
            
            # 创建Explanation对象用于瀑布图
            exp = shap.Explanation(
                values=shap_values_array[0],  # 第一个样本的SHAP值
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=shap_df.iloc[0].values,
                feature_names=shap_df.columns.tolist()
            )
            
            # 绘制瀑布图
            shap.plots.waterfall(exp, max_display=8, show=False)
            plt.title(f"SHAP瀑布图 - PI预测概率: {probability:.2f}%", fontsize=12, pad=20)
            plt.tight_layout()
            
            # 保存瀑布图为图像
            buf_waterfall = BytesIO()
            plt.savefig(buf_waterfall, format="png", bbox_inches="tight", dpi=300)
            plt.close()
            
            # 重置缓冲区位置
            buf_force.seek(0)
            buf_waterfall.seek(0)
            
            # 显示SHAP解释图
            st.subheader("模型解释")
            st.markdown("以下图表显示了各个特征变量对预测结果的贡献程度：")
            
            # 创建两列显示两个图
            col_force, col_waterfall = st.columns(2)
            
            with col_force:
                st.markdown("#### SHAP力图")
                st.image(buf_force, use_column_width=True)
                st.caption("力图显示了每个特征如何将模型输出从基准值推动到最终预测值")
            
            with col_waterfall:
                st.markdown("#### SHAP瀑布图")
                st.image(buf_waterfall, use_column_width=True)
                st.caption("瀑布图显示了每个特征对预测的累积贡献")
            
            # 添加特征影响分析
            st.subheader("特征影响分析")
            
            # 计算每个特征的SHAP值贡献
            feature_shap = {}
            for i, feature in enumerate(shap_df.columns):
                feature_shap[feature] = shap_values_array[0][i]
            
            # 按绝对贡献值排序
            sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # 显示前5个最重要的特征
            st.markdown("**对预测影响最大的特征：**")
            for feature, shap_value in sorted_features[:5]:
                direction = "增加" if shap_value > 0 else "降低"
                color = "red" if shap_value > 0 else "green"
                st.markdown(f"- **{feature}**: <span style='color:{color}'>{direction}PI风险</span> (影响值: {shap_value:.4f})", 
                           unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"生成模型解释图时出错: {str(e)}")
            st.info("请确保SHAP库已正确安装，并且模型支持SHAP解释。")

# 添加侧边栏信息
with st.sidebar:
    st.header("关于本应用")
    st.markdown(f"""
    ### 开发信息
    - **开发单位**: {AUTHOR_INFO["institution"]}
    - **作者**: {AUTHOR_INFO["author"]}
    
    ### 模型信息
    - **算法**: Logistic Regression (逻辑回归)
    - **预测目标**: 压力性损伤(PI)风险
    - **应用场景**: 临床风险评估
    
    ### 使用说明
    1. 在表单中输入患者临床指标
    2. 点击"开始预测"按钮
    3. 查看预测结果和模型解释
    
    ### 注意事项
    - 本工具仅供临床参考
    - 实际诊疗请结合临床判断
    - 如有疑问请咨询专业医师
    """)

# 添加特征缩写说明
with st.sidebar.expander("特征缩写说明"):
    st.markdown("""
    | 缩写 | 全称 | 描述 |
    |------|------|------|
    | FCTI | FCTI总分 | 功能沟通测试工具总分 |
    | Age | 年龄 | 患者年龄（岁） |
    | Ser | 血清白蛋白 | 血清白蛋白水平 |
    | Fra | 骨折类型 | 骨折的具体类型 |
    | Air | 气垫床/充气床垫 | 是否使用气垫床 |
    | Com | 合并症数量 | 患者合并症的数量 |
    | PCAT | PCAT总分 | 患者照顾者评估工具总分 |
    | Mlu | 多发性骨折 | 是否有多发性骨折 |
    """)

# 添加页脚
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        临床决策支持工具 • {AUTHOR_INFO["institution"]} • {AUTHOR_INFO["author"]} • 仅供参考
    </div>
    """, 
    unsafe_allow_html=True
)

# 添加SHAP图例说明
with st.expander("如何解读SHAP图"):
    st.markdown("""
    ### SHAP力图解读
    - **红色箭头**：增加PI风险的因素
    - **蓝色箭头**：降低PI风险的因素  
    - **箭头长度**：表示该因素影响程度的大小
    - **基准值**：模型在训练数据上的平均预测值
    - **输出值**：当前患者的预测概率
    
    ### SHAP瀑布图解读
    - **从上到下**：显示了每个特征如何将预测值从基准值推到最终预测值
    - **条形长度**：表示每个特征的影响大小
    - **红色条形**：正向影响（增加风险）
    - **蓝色条形**：负向影响（降低风险）
    - **底部值**：最终预测概率
    """)
