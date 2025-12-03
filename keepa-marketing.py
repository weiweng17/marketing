# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from datetime import datetime
import io
from collections import Counter
import re
import ai_analysis
import numpy as np
import scoring_logic
import copy
# --- 页面基础设置 ---
st.set_page_config(
    page_title="亚马逊深度选品分析 V9.0 Ultimate",
    layout="wide",
    page_icon="🦁",
    initial_sidebar_state="expanded"
)

# --- 全局图表配置 (保持原版配置) ---
DOWNLOAD_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'market_analysis_chart_awai',
        'height': 800,
        'width': 1200,
        'scale': 2
    }
}

# 强制白底模板，解决导出黑白问题
TEMPLATE_THEME = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Pastel

# --- 全局样式 (完全还原原版 CSS) ---
st.markdown("""
<style>
    .main-header {font-size: 24px; font-weight: bold; color: #2E4053; margin-bottom: 20px;}
    .metric-box {background-color: #F4F6F7; padding: 15px; border-radius: 8px; border-left: 5px solid #3498DB;}
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0;}
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-top: 3px solid #3498DB;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. 数据清洗函数 (逻辑严格还原 V3.4)
# ==========================================
@st.cache_data
def load_data(file):
    try:
        # 1. 读取文件
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file, encoding='gbk')
                except:
                    df = pd.read_csv(file, encoding='gb18030')
        else:
            df = pd.read_excel(file)

        # 2. 表头清洗 (去除前后空格)
        df.columns = df.columns.str.strip()

        # 3. 货币与数字清洗
        cols_to_clean = ['月销售额($)', '价格($)', 'FBA($)', '子体销售额($)', '买家运费($)']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d.-]', '', x))
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 4. 百分比清洗
        percent_cols = ['毛利率', '留评率', '月销量增长率']
        for col in percent_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 5. 整数清洗
        int_cols = ['月销量', '评分数', '上架天数', '变体数']
        for col in int_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # 6. 日期清洗
        if '上架时间' in df.columns:
            df['上架时间'] = pd.to_datetime(df['上架时间'], errors='coerce')
            current_time = pd.Timestamp.now()
            df['计算上架天数'] = (current_time - df['上架时间']).dt.days
            if '上架天数' not in df.columns or df['上架天数'].sum() == 0:
                df['上架天数'] = df['计算上架天数'].fillna(0).astype(int)
            df['上架月份'] = df['上架时间'].dt.month_name()
            df['是否新品'] = df['上架天数'].apply(lambda x: '新品 (<90天)' if x <= 90 else '老品')

        # 7. 文本填充
        text_cols = ['品牌', '大类目', '配送方式', 'BuyBox类型', '商品标题']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
            elif col in ['品牌', '大类目']:
                df[col] = 'Unknown'

        # ==========================================
        # 🖼️ 智能图片列识别 (V12.0 增强版)
        # ==========================================
        found_img_col = None

        # 策略A: 模糊匹配列名 (忽略大小写)
        potential_cols = [c for c in df.columns if
                          any(k in c.lower() for k in ['image', 'img', 'photo', '主图', '图片'])]

        # 策略B: 内容检测 (如果列名匹配到了，检查内容是否像URL)
        for col in potential_cols:
            # 取第一条非空数据检查
            sample = df[col].dropna().astype(str).iloc[0] if not df[col].dropna().empty else ""
            if sample.startswith('http'):
                found_img_col = col
                break

        # 赋值逻辑：先存为原始列名，后续在 main 函数中处理
        if found_img_col:
            df['__Auto_Detected_Image_Col__'] = df[found_img_col]
        else:
            # 如果没找到，尝试用 ASIN 构造
            if 'ASIN' in df.columns:
                df['__Auto_Detected_Image_Col__'] = df['ASIN'].apply(
                    lambda
                        asin: f"https://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&Format=_SL250_&ASIN={str(asin).strip()}&MarketPlace=US&ID=AsinImage&WS=1&ServiceVersion=20070822"
                )
            else:
                df['__Auto_Detected_Image_Col__'] = None

        return df

    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return None


# --- 参数解析器 ---
@st.cache_data
def parse_detailed_params(df):
    """保留正则优化版本，因为确实比循环快"""
    if '详细参数' not in df.columns:
        return df
    df_new = df.copy()

    def extract_params(text):
        if pd.isna(text) or text == '': return {}
        pattern = r'([^:|]+):([^|]+)'
        matches = re.findall(pattern, str(text))
        return {k.strip(): v.strip() for k, v in matches}

    parsed_series = df_new['详细参数'].apply(extract_params)
    params_df = pd.DataFrame(parsed_series.tolist())
    params_df.columns = [f"参数_{c}" for c in params_df.columns]
    # 过滤稀疏列
    threshold = len(df) * 0.05
    params_df = params_df.dropna(thresh=threshold, axis=1)
    return pd.concat([df_new, params_df], axis=1)


# --- 关键词分析 ---
@st.cache_data
def analyze_keywords(df, top_n=30):
    if '商品标题' not in df.columns: return None
    text = " ".join(df['商品标题'].dropna().astype(str).tolist()).lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords = set(
        ['the', 'for', 'and', 'with', 'of', 'to', 'in', 'a', 'on', 'at', 'pack', 'pcs', 'set', 'new', 'black', 'white',
         'unknown', 'nan', 'generic'])
    words = [w for w in text.split() if w not in stopwords and not w.isdigit() and len(w) > 2]
    return pd.DataFrame(Counter(words).most_common(top_n), columns=['关键词', '出现频次'])


# ==========================================
# 2. HTML 报告生成 (结构还原 V3.4 + 技术使用 V8.0)
# ==========================================

def generate_interactive_html_report(df, charts_data, analysis_data, target_attr=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 准备图表 HTML
    charts_html = ""
    for chart_name, fig in charts_data.items():
        # --- 核心修复 1: 强制冻结尺寸 ---
        fig.update_layout(
            template=TEMPLATE_THEME,
            paper_bgcolor='white',
            plot_bgcolor='white',
            width=1200,  # 强制宽度
            height=600,  # 强制高度
            autosize=False,  # 关闭自适应
            margin=dict(l=60, r=60, t=80, b=60)  # 预留边距防止文字被切
        )

        # --- 核心修复 2: 嵌入 JS ---
        # include_plotlyjs=True 会把几MB的引擎直接写进文件，解决国内加载CDN失败的问题
        # full_html=False 只生成 div 部分，我们在后面自己拼接 HTML 骨架
        chart_div = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',  # 先尝试cdn减小体积，如果还是不行，请改成 True (注意大小写)
            config={'responsive': False, 'displayModeBar': True}  # 关闭响应式，防止变形
        )

        # 如果想彻底解决白屏，请把上面 include_plotlyjs='cdn' 改为 include_plotlyjs=True
        # 这样文件会变大(3MB+)，但绝对能显示。
        if 'cdn' in chart_div:
            # 这是一个保险逻辑，如果你想用离线版，请直接用下面这句覆盖上面的 chart_div 生成逻辑
            chart_div = fig.to_html(full_html=False, include_plotlyjs=True, config={'responsive': False})

        charts_html += f"""
        <div class="chart-section">
            <h3>{chart_name}</h3>
            <div class="chart-container" style="width:1200px; height:600px; margin:0 auto;">
                {chart_div}
            </div>
        </div>
        """

    # 品牌表格逻辑
    brand_rows = ""
    total_rev = analysis_data['total_revenue']
    for brand, revenue in analysis_data['top_brands'].items():
        market_share = (revenue / total_rev) * 100 if total_rev > 0 else 0
        brand_rows += f"""
        <tr>
            <td><strong>{brand}</strong></td>
            <td>${revenue:,.0f}</td>
            <td>{market_share:.1f}%</td>
        </tr>
        """

    # HTML 骨架 (移除了 <head> 里的 script 引用，因为我们已经在图表里嵌入了)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>亚马逊交互式市场分析报告 - {timestamp}</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 0; padding: 20px; background-color: #f4f6f9; }}
            .container {{ max-width: 1280px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; padding-bottom: 30px; border-bottom: 2px solid #eee; margin-bottom: 30px; }}
            .header h1 {{ color: #2c3e50; margin: 0; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }}
            .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e9ecef; }}
            .metric-val {{ font-size: 24px; font-weight: bold; color: #2980b9; margin: 10px 0; }}
            .section-title {{ font-size: 20px; border-left: 5px solid #3498DB; padding-left: 15px; margin: 40px 0 20px 0; color: #34495e; }}
            .chart-section {{ margin-bottom: 50px; border: 1px solid #eee; padding: 20px; border-radius: 8px; background: #fff; overflow-x: auto; }}
            .brand-table {{ width: 100%; border-collapse: collapse; }}
            .brand-table th, .brand-table td {{ padding: 12px; border-bottom: 1px solid #eee; text-align: left; }}
            .brand-table th {{ background-color: #f8f9fa; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 亚马逊市场深度分析报告</h1>
                <p>生成时间: {timestamp} | 产品数: {analysis_data['total_products']}</p>
            </div>

            <div class="metric-grid">
                <div class="metric-card"><div>平均月销量</div><div class="metric-val">{analysis_data['avg_monthly_sales']:.0f}</div></div>
                <div class="metric-card"><div>平均月销售额</div><div class="metric-val">${analysis_data['avg_monthly_revenue']:,.0f}</div></div>
                <div class="metric-card"><div>平均增长率</div><div class="metric-val">{analysis_data['avg_growth_rate']:.1f}%</div></div>
                <div class="metric-card"><div>头部品牌份额</div><div class="metric-val">Top 5</div></div>
            </div>

            <h2 class="section-title">📊 交互式图表分析</h2>
            {charts_html}

            <h2 class="section-title">🏆 品牌数据</h2>
            <table class="brand-table">
                <thead><tr><th>品牌</th><th>销售额</th><th>占比</th></tr></thead>
                <tbody>{brand_rows}</tbody>
            </table>

            <div style="text-align:center; margin-top:50px; color:#999; font-size:12px;">Generated by Amazon Ultimate Tool</div>
        </div>
    </body>
    </html>
    """
    return html_content


def create_download_link(content, filename, text):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="{filename}" style="display:block; width:100%; padding:12px; text-align:center; background:#27ae60; color:white; text-decoration:none; border-radius:5px; font-weight:bold; margin-top:10px;">📥 {text}</a>'
    return href


# ==========================================
# [新增模块] 表格合并处理函数
# ==========================================
def process_and_merge_tables(file_main, file_keepa):
    """
    处理合并逻辑：
    1. 读取两个文件
    2. 填充 Keepa 表的 Parent ASIN
    3. 以 ASIN 为键进行合并
    """
    try:
        # 1. 读取主表 (JS/H10)
        if file_main.name.endswith('.csv'):
            df_main = pd.read_csv(file_main)
        else:
            df_main = pd.read_excel(file_main)

        # 2. 读取 Keepa 表
        if file_keepa.name.endswith('.csv'):
            df_keepa = pd.read_csv(file_keepa)
        else:
            df_keepa = pd.read_excel(file_keepa)

        # 3. 基础清洗 ASIN (去空格)
        if 'ASIN' in df_main.columns:
            df_main['ASIN'] = df_main['ASIN'].astype(str).str.strip()

        if 'ASIN' in df_keepa.columns:
            df_keepa['ASIN'] = df_keepa['ASIN'].astype(str).str.strip()
        else:
            return None, "错误：Keepa表中找不到 'ASIN' 列"

        # 4. [核心需求] 填充 Parent ASIN
        # 逻辑：如果 Parent ASIN 为空，则填入 ASIN
        target_col = 'Parent ASIN'
        if target_col in df_keepa.columns:
            # 将空白字符替换为 NaN
            df_keepa[target_col] = df_keepa[target_col].replace(r'^\s*$', np.nan, regex=True)
            # 填充
            df_keepa[target_col] = df_keepa[target_col].fillna(df_keepa['ASIN'])

        # 5. 合并 (Left Join, 保留主表所有数据)
        merged_df = pd.merge(
            df_main,
            df_keepa,
            on='ASIN',
            how='left',
            suffixes=('', '_Keepa')  # 主表列名不变，Keepa重复列加后缀
        )

        return merged_df, "Success"

    except Exception as e:
        return None, str(e)


def to_excel_bytes(df):
    """将DataFrame转为二进制流用于下载"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Merged_Data')
    output.seek(0)
    return output
# ==========================================
# 3. 主程序
# ==========================================
def main():
    import random

    DATA_INSIGHTS = [
        "📉 **当竞品纷纷涌入某个热销品类，那往往意味着利润空间正在被迅速摊薄。** —— 警惕红海虚假繁荣",
        "🔍 **当一款产品的搜索增长率，远高于其加购与转化率的增长，那说明市场好奇心的背后，是购买决策的犹豫。** —— 痛点未被满足",
        "💰 **当广告投入增长快于销售增长，那说明 ROI 的假象正在掩盖利润侵蚀。** —— 关注净利而非 GMV",
        "📦 **库存的本质是资金的占用。缺货不仅仅是损失销量，更是将辛苦打下的市场份额拱手让人。** —— Keepa 缺货监控",
        "⭐ **差评往往比好评更有价值。好评告诉你用户为什么买，差评告诉你用户为什么走。** —— 改良产品的金钥匙",
        "⚖️ **价格战没有赢家，只有幸存者。依靠低价获取的客户，忠诚度也是最低的。** —— 品牌护城河",
        "🌊 **不要试图创造需求，要去发现那些已经存在但未被满足的需求。** —— 选品底层逻辑",
        "📊 **数据本身不产生价值，对数据背后逻辑的解读和行动才产生价值。** —— 拒绝数据焦虑",
        "🎨 **在标品市场拼效率，在非标品市场拼审美。你的产品属性决定了你的核心竞争力。** —— 属性分析",
        "🚀 **最好的防守是进攻。当你的 listing 长期不更新，就是在给竞品弯道超车的机会。** —— 保持活跃度"
    ]
    # 🟢【插入结束】🟢
    # 侧边栏
    st.sidebar.title("🛠️ 分析控制台")
    uploaded_file = st.sidebar.file_uploader("上传市场调研数据 (Excel/CSV)", type=['xlsx', 'csv'], key="main_analysis_upload")
    with st.sidebar.expander("🧩 数据预处理 (合并表格)", expanded=False):
        st.caption("功能：合并运营表与Keepa表，并修补Parent ASIN")
        f1 = st.file_uploader("1. 上传主表", type=['xlsx', 'csv'], key="m1")
        f2 = st.file_uploader("2. 上传Keepa表", type=['xlsx', 'csv'], key="m2")

        if f1 and f2:
            if st.button("🚀 开始合并", key="btn_merge"):
                with st.spinner("处理中..."):
                    res_df, msg = process_and_merge_tables(f1, f2)
                    if res_df is not None:
                        st.success(f"合并成功! 共 {len(res_df)} 行")
                        st.download_button("📥 下载结果表格", data=to_excel_bytes(res_df),
                                           file_name=f"Merged_{datetime.now().strftime('%H%M%S')}.xlsx")
                    else:
                        st.error(f"失败: {msg}")

    st.sidebar.markdown("---")  # 加个分割线好看点

    # 还原：底部签名
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Data Analysis Tool | 阿伟出品")

    if uploaded_file:
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            df = df_raw.copy()

            with st.sidebar.expander("🖼️ 图片显示设置", expanded=True):
                # 获取所有列名
                all_cols = df.columns.tolist()
                # 排除掉我们内部生成的列
                clean_cols = [c for c in all_cols if c != '__Auto_Detected_Image_Col__']

                # 下拉框：默认选择 "自动检测"
                img_option = st.selectbox(
                    "选择包含图片的列:",
                    options=["⚡ 自动检测 / ASIN构造"] + clean_cols,
                    help="如果图片显示失败，请在此处手动选择你的表格中包含图片链接的那一列"
                )

                # 逻辑判断
                if img_option == "⚡ 自动检测 / ASIN构造":
                    # 使用 load_data 中自动生成的列
                    df['Product_Img'] = df.get('__Auto_Detected_Image_Col__')
                else:
                    # 使用用户手动指定的列
                    st.success(f"已指定: {img_option}")
                    df['Product_Img'] = df[img_option]
            # 还原：详细参数解析开关
            with st.sidebar.expander("🔧 参数解析设置", expanded=False):
                df = parse_detailed_params(df)
                param_cols = [c for c in df.columns if c.startswith('参数_')]
                if param_cols:
                    st.success(f"✅ 已解析 {len(param_cols)} 个参数")
                else:
                    st.info("未检测到'详细参数'列或格式不匹配")

            # 还原：数据质量诊断
            with st.sidebar.expander("🔍 数据质量诊断", expanded=True):
                st.write(f"- 总行数: {len(df)}")
                if 'ASIN' in df.columns:
                    dups = df['ASIN'].duplicated().sum()
                    if dups > 0:
                        st.error(f"发现 {dups} 个重复 ASIN")
                    else:
                        st.success("ASIN 无重复")
                if '月销量增长率' in df.columns:
                    missing_growth = df['月销量增长率'].isna().sum()
                    st.write(f"- 缺失增长率: {missing_growth} 条")

            # 侧边栏筛选
            brands = st.sidebar.multiselect("品牌筛选", sorted(df['品牌'].unique()))
            if brands:
                df = df[df['品牌'].isin(brands)]

            # 属性选择
            all_cols = df.columns.tolist()

            # 1. 提取解析出来的详细参数
            param_cols = [c for c in all_cols if c.startswith('参数_')]

            # 2. 定义白名单关键词 (⚡️ 全面扩充版：包含 Keepa 物理规格)
            target_keywords = [
                # === A. 基础变体属性 ===
                '颜色', 'Color', '材质', 'Material', '尺寸', 'Size', 'Style', 'Pattern',

                # === B. 运营核心字段 (Keepa/Amazon) ===
                'Group', 'Product Group',  # 产品分组
                'Manufacturer', 'Brand',  # 制造商
                'Model',  # 型号
                'Binding',  # 包装形式
                'Type',  # 类型
                'Is FBA', 'Fulfillment',  # 配送方式
                'Status',  # 状态
                'Department',  # 适用人群
                'Is Prime',  # Prime 标记

                # === C. 物理规格 (新增部分 📏) ===
                # 英文关键词 (Keepa 原生字段通常包含这些)
                'Length', 'Width', 'Height', 'Weight', 'Dimension',
                # 中文关键词 (如果你汉化过表头)
                '长', '宽', '高', '重',
                # Keepa 特有前缀 (用于匹配 'Item: Length (cm)', 'Package: Weight (g)' 等)
                'Item:', 'Package:'
            ]

            # 3. 扫描列名
            keyword_cols = [c for c in all_cols if any(k in c for k in target_keywords)]

            # 4. 合并去重并排序
            valid_attrs = sorted(list(set(param_cols + keyword_cols)))

            # 5. 渲染下拉框
            target_attr = st.sidebar.selectbox("🎯 选择重点分析属性", valid_attrs) if valid_attrs else None

            # --- 计算核心指标 (严格遵守 V3.4 逻辑) ---
            # 还原：增长率乘以100
            avg_growth_val = df['月销量增长率'].mean() * 100 if '月销量增长率' in df.columns else 0

            # --- 主界面 ---
            st.title("🚀 亚马逊全维度市场扫描报告")
            # 🟢【在这里插入展示代码】🟢
            if 'DATA_INSIGHTS' in locals():
                chosen_quote = random.choice(DATA_INSIGHTS)
                st.markdown(f"""
                          <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3498DB; margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                              <span style="font-size: 16px; color: #2C3E50; font-family: 'Microsoft YaHei';">
                                  💡 <strong>Deep Insight:</strong> {chosen_quote}
                              </span>
                          </div>
                          """, unsafe_allow_html=True)
            # 🟢【插入结束】🟢

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("平均月销量", f"{df['月销量'].mean():.0f} 件")
            col_m2.metric("平均月销售额", f"${df['月销售额($)'].mean():,.0f}")
            col_m3.metric("平均增长率", f"{avg_growth_val:.1f}%")  # 使用乘以100后的值
            top_cat = df['大类目'].mode()[0] if '大类目' in df.columns else "未知"
            col_m4.metric("最热销类目", top_cat)

            # 收集图表
            export_charts = {}

            # 模块 1: 机会矩阵 + 排行榜
            st.header("1. 市场机会扫描 (Market Opportunity)")
            # === ✨ 新增：Top 3 冠军画廊 ===
            if '月销售额($)' in df.columns and 'Product_Img' in df.columns:
                # 🔽 修改点：按 '月销售额($)' 降序排列，取前3
                top3 = df.sort_values('月销售额($)', ascending=False).head(3).reset_index()

                with st.expander(f"🏆 点击展开：市场销售额(GMV) Top 3 冠军产品画廊", expanded=False):
                    g1, g2, g3 = st.columns(3)

                    # 辅助函数：渲染卡片 (保留之前的防报错逻辑)
                    def render_card(col, row, rank):
                        with col:
                            # 增加皇冠图标区分名次
                            crowns = {1: "🥇", 2: "🥈", 3: "🥉"}
                            st.markdown(f"#### {crowns.get(rank, '')} No.{rank}")

                            # 图片渲染 (带类型检查)
                            img_url = row.get('Product_Img')
                            if isinstance(img_url, str) and len(img_url) > 5:
                                try:
                                    st.image(img_url, width=150)
                                except:
                                    st.warning("图片加载失败")
                            else:
                                st.info("🖼️ 暂无图片")

                            # 显示核心数据 (销售额加粗显示)
                            st.markdown(f"""
                                      - **ASIN**: `{row['ASIN']}`
                                      - **月收**: **${row['月销售额($)']:,.0f}**
                                      - **销量**: {row['月销量']} 件
                                      - **价格**: ${row['价格($)']:.2f}
                                      """)

                            if '商品标题' in row:
                                short_title = str(row['商品标题'])[:50] + "..."
                                st.caption(short_title)

                    # 依次渲染
                    if len(top3) >= 1: render_card(g1, top3.iloc[0], 1)
                    if len(top3) >= 2: render_card(g2, top3.iloc[1], 2)
                    if len(top3) >= 3: render_card(g3, top3.iloc[2], 3)
            # ==================================
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown("#### 🔮 增长率 vs 销量矩阵 (ASIN定位)")
                st.info("💡 **操作**: 悬停查看 ASIN，点击右上角相机下载。")

                if '月销量增长率' in df.columns:
                    # 还原：数据准备逻辑
                    df_display = df.copy()
                    df_display['月销量增长率_显示'] = df_display['月销量增长率'] * 100  # 还原 * 100

                    # 清洗 NaN 以防导出崩溃
                    df_display['月销售额($)'] = df_display['月销售额($)'].fillna(0)
                    df_display['月销量'] = df_display['月销量'].fillna(0)
                    # 气泡大小修正：确保最小可见度，且不为0
                    df_display['BubbleSize'] = df_display['月销售额($)'].apply(
                        lambda x: max(x, 100) if pd.notnull(x) else 100)

                    # 找到生成 fig_matrix 的地方
                    fig_matrix = px.scatter(
                        df_display,
                        x="月销量增长率_显示",
                        y="月销量",
                        color="上架天数",  # 原始：颜色随上架天数变化
                        size="BubbleSize",  # 原始：大小随销售额变化
                        hover_name="ASIN",
                        # 显式指定 hover 数据
                        hover_data={
                            "BubbleSize": False,
                            "商品标题": True,
                            "品牌": True,
                            "价格($)": True,
                            "月销售额($)": ":,.0f"
                        },
                        title="产品潜力四象限分析 - 增长率 vs 销量",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        template=TEMPLATE_THEME
                    )

                    fig_matrix.add_hline(y=df['月销量'].mean(), line_dash="dash", line_color="red",
                                         annotation_text="平均销量")
                    fig_matrix.add_vline(x=avg_growth_val, line_dash="dash", line_color="blue",
                                         annotation_text="平均增长")
                    fig_matrix.update_xaxes(title_text="月销量增长率 (%)")

                    # 修正悬停显示，使其显示真实数据而非气泡半径
                    fig_matrix.update_traces(
                        hovertemplate="<b>%{hovertext}</b><br>增长率: %{x:.1f}%<br>销量: %{y}<br>销售额: $%{customdata[4]:,.0f}<br>品牌: %{customdata[2]}")

                    st.plotly_chart(fig_matrix, width="stretch", config=DOWNLOAD_CONFIG)
                    export_charts["📈 产品潜力四象限分析"] = fig_matrix

            with c2:
                st.markdown("#### 📸 视觉化飙升榜 (Top 20)")
                if '月销量增长率' in df.columns:
                    # 准备数据
                    rank_df = df.copy()
                    rank_df['月销量增长率'] = rank_df['月销量增长率'] * 100
                    rank_df = rank_df.sort_values('月销量增长率', ascending=False).head(20)

                    # 选取展示列 (把图片列放到第一位)
                    display_cols = ['Product_Img', 'ASIN', '月销量', '月销量增长率', '价格($)']
                    # 确保列存在
                    display_cols = [c for c in display_cols if c in rank_df.columns]

                    st.dataframe(
                        rank_df[display_cols],
                        hide_index=True,
                        column_config={
                            "Product_Img": st.column_config.ImageColumn(
                                "产品主图",
                                help="点击查看大图",
                                width="small"  # 设置图片大小
                            ),
                            "ASIN": st.column_config.TextColumn("ASIN", width="small"),
                            "月销量": st.column_config.ProgressColumn(
                                "月销量",
                                format="%d",
                                min_value=0,
                                max_value=int(df['月销量'].max())
                            ),
                            "月销量增长率": st.column_config.NumberColumn(
                                "增长率",
                                format="%.1f%%"
                            ),
                            "价格($)": st.column_config.NumberColumn(
                                "价格",
                                format="$%.2f"
                            )
                        },
                        height=600  # 稍微调高一点，展示图片需要空间
                    )

            st.divider()

            # Tab 页结构
            tabs = st.tabs(["🧬 属性深度分析", "🏆 品牌与时间", "📦 卖家与新品", "🗝️ NLP与高级统计", "📊 Keepa深度洞察"])

            # Tab 1: 属性 (优化版：兼容数字与文本)
            with tabs[0]:
                if target_attr:
                    st.header(f"2. 属性深度分析: {target_attr}")

                    df_analysis = df.copy()

                    # === 🟢 新增逻辑：判断是数字列还是文本列 ===
                    # 尝试将列转换为数字，无法转换的变为 NaN
                    numeric_series = pd.to_numeric(df_analysis[target_attr], errors='coerce')

                    # 2. 智能判断：是数字还是文本？
                    # 逻辑：如果转换后包含了有效数字 (notna)，我们就认为它是数字列；
                    #      如果转换后全是 NaN，说明它是纯文本列 (如颜色/品牌)，我们保持原样不动。
                    if numeric_series.notna().any():
                        df_analysis[target_attr] = numeric_series
                        is_numeric = True
                    else:
                        # 转换失败，说明是文本，保持原样
                        is_numeric = False

                    # 1. 数据预处理 (如果是数字，进行取整，防止太散)
                    if is_numeric:
                        # 逻辑：如果是尺寸/重量，保留1位小数或取整，这样相近的尺寸会合并成一组
                        # 这里我们统一保留 0 位小数 (即取整)，图表会更清晰。
                        # 如果需要更精细，可以把 round(0) 改为 round(1)
                        df_analysis[target_attr] = df_analysis[target_attr].round(0)

                        # 过滤掉 0 或异常值（可选）
                        df_analysis = df_analysis[df_analysis[target_attr] > 0]

                    # 还原：增长率计算
                    if '月销量增长率' in df_analysis.columns:
                        df_analysis['月销量增长率_显示'] = df_analysis['月销量增长率'] * 100
                    else:
                        df_analysis['月销量增长率_显示'] = 0

                    # 2. 聚合统计
                    attr_group = df_analysis.groupby(target_attr).agg({
                        '月销量': 'sum',
                        '月销售额($)': 'sum',
                        '月销量增长率_显示': 'mean',
                        '价格($)': 'mean',
                        'ASIN': 'count'
                    }).reset_index()

                    # === 🟢 新增逻辑：排序策略 ===
                    if is_numeric:
                        # 策略 A: 如果是数字，按【数值本身】从小到大排序 (解决折线乱跑问题)
                        top_attrs = attr_group.sort_values(target_attr, ascending=True)
                        # 如果分组太多，图表会很卡，限制显示前 50 个区间
                        if len(top_attrs) > 50:
                            # 这种情况下，我们还是取销售额 Top 50，然后再按数值排序
                            top_attrs = attr_group.sort_values('月销售额($)', ascending=False).head(50)
                            top_attrs = top_attrs.sort_values(target_attr, ascending=True)
                    else:
                        # 策略 B: 如果是文本，按【销售额】从高到低排序 (看谁卖得好)
                        top_attrs = attr_group.sort_values('月销售额($)', ascending=False).head(15)

                    top_vals = top_attrs[target_attr].tolist()

                    # 3. 绘图 (保持原有逻辑，但数据源已优化)
                    t1, t2 = st.columns(2)
                    with t1:
                        # 组合图
                        fig_combo = go.Figure()
                        fig_combo.add_trace(
                            go.Bar(x=top_attrs[target_attr], y=top_attrs['月销售额($)'], name='月销售额($)',
                                   marker_color='#3498DB'))
                        fig_combo.add_trace(
                            go.Scatter(x=top_attrs[target_attr], y=top_attrs['月销量'], name='月销量', yaxis='y2',
                                       mode='lines+markers', line=dict(color='#E74C3C')))

                        fig_combo.update_layout(
                            title=f"{target_attr} 销售额(柱) 与 销量(折线)",
                            yaxis=dict(title='销售额 ($)'),
                            yaxis2=dict(title='销量 (件)', overlaying='y', side='right'),
                            template=TEMPLATE_THEME,
                            legend=dict(orientation="h", y=1.1),
                            # 如果是数字，强制 X 轴为类别型，防止 Plotly 自动补全中间的空缺数字
                            xaxis=dict(type='category')
                        )
                        st.plotly_chart(fig_combo, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts[f"💰 {target_attr} 销售分析"] = fig_combo

                    with t2:
                        # 价格分布
                        filtered_attr_df = df_analysis[df_analysis[target_attr].isin(top_vals)]

                        # 🟢 关键：如果是数字，需要强制排序，否则箱线图顺序也会乱
                        if is_numeric:
                            filtered_attr_df = filtered_attr_df.sort_values(target_attr)

                        fig_box = px.box(filtered_attr_df, x=target_attr, y="价格($)", color=target_attr,
                                         title=f"{target_attr} 价格分布", template=TEMPLATE_THEME)
                        fig_box.update_layout(showlegend=False, xaxis=dict(type='category'))
                        st.plotly_chart(fig_box, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts[f"💰 {target_attr} 价格分析"] = fig_box

                    # 增长率 Bar
                    fig_growth = px.bar(top_attrs, x=target_attr, y="月销量增长率_显示", color="月销量增长率_显示",
                                        color_continuous_scale="RdYlGn", title=f"🚀 {target_attr} 增长趋势",
                                        template=TEMPLATE_THEME)
                    fig_growth.update_yaxes(title_text="月销量增长率 (%)")
                    fig_growth.update_layout(xaxis=dict(type='category'))
                    fig_growth.update_traces(hovertemplate='%{x}<br>增长率: %{y:.1f}%')
                    st.plotly_chart(fig_growth, width="stretch", config=DOWNLOAD_CONFIG)
                    export_charts[f"🚀 {target_attr} 增长分析"] = fig_growth

            # Tab 2: 品牌 (还原逻辑)
            with tabs[1]:
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown("#### 品牌市场占有率 Top 10")
                    # 修复：导出全是1的问题 -> 传递聚合后的数据
                    brand_share = df.groupby('品牌')['月销售额($)'].sum().reset_index().sort_values('月销售额($)',
                                                                                                    ascending=False).head(
                        10)
                    fig_pie = px.pie(brand_share, values='月销售额($)', names='品牌', hole=0.4,
                                     color_discrete_sequence=COLOR_SEQUENCE, template=TEMPLATE_THEME)
                    st.plotly_chart(fig_pie, width="stretch", config=DOWNLOAD_CONFIG)
                    export_charts["🏆 品牌市场占有率"] = fig_pie

                with b2:
                    st.markdown("#### 📅 爆款通常在几月上架？")
                    if '上架月份' in df.columns:
                        high_sales_df = df[df['月销量'] > df['月销量'].mean()]
                        month_counts = high_sales_df['上架月份'].value_counts().reset_index()
                        month_counts.columns = ['月份', '数量']
                        fig_month = px.bar(month_counts, x='月份', y='数量', title="热销品上架月份分布",
                                           template=TEMPLATE_THEME)
                        st.plotly_chart(fig_month, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts["📅 上架月份分析"] = fig_month

                st.markdown("#### 🗓️ 品牌上架时间 vs 销售额 (寻找常青树)")
                if '上架时间' in df.columns:
                    # 修复：气泡大小数据清洗
                    t_df = df.copy()
                    t_df['月销量'] = t_df['月销量'].fillna(0)
                    t_df['Size'] = t_df['月销量'].apply(lambda x: max(x, 10))  # 保证不为0

                    fig_time = px.scatter(
                        t_df, x="上架时间", y="月销售额($)", color="品牌", size="Size",
                        hover_name="ASIN",
                        # 显式指定 hover data
                        hover_data={'Size': False, '月销量': True, '月销售额($)': ':,.0f'},
                        title="上架时间分布：谁是老牌霸主？",
                        template=TEMPLATE_THEME, size_max=60
                    )
                    st.plotly_chart(fig_time, width="stretch", config=DOWNLOAD_CONFIG)
                    export_charts["📅 上架时间分析"] = fig_time

            # Tab 3: 卖家 (还原逻辑)
            with tabs[2]:
                col_last1, col_last2, col_last3 = st.columns(3)
                with col_last1:
                    if '配送方式' in df.columns:
                        fig_fba = px.pie(df, names='配送方式', title='配送方式占比',
                                         color_discrete_sequence=px.colors.qualitative.Set2, template=TEMPLATE_THEME)
                        st.plotly_chart(fig_fba, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts["👥 配送方式占比"] = fig_fba
                with col_last2:
                    if 'BuyBox类型' in df.columns:
                        fig_bb = px.pie(df, names='BuyBox类型', title='卖家类型占比',
                                        color_discrete_sequence=px.colors.qualitative.Set3, template=TEMPLATE_THEME)
                        st.plotly_chart(fig_bb, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts["👥 卖家类型占比"] = fig_bb
                with col_last3:
                    if '是否新品' in df.columns:
                        new_share = df.groupby('是否新品')['月销售额($)'].sum().reset_index()
                        fig_new = px.pie(new_share, values='月销售额($)', names='是否新品', title='新品市场占有率',
                                         color='是否新品',
                                         color_discrete_map={'新品 (<90天)': '#2ECC71', '老品': '#95A5A6'},
                                         template=TEMPLATE_THEME)
                        st.plotly_chart(fig_new, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts["👥 新品市场占有率"] = fig_new

            # Tab 4: 高级 (保留新增功能，但修复导出)
            with tabs[3]:
                st.markdown("#### 🗝️ NLP 标题高频词")
                kw_df = analyze_keywords(df)
                if kw_df is not None:
                    fig_kw = px.bar(kw_df.head(20), x='出现频次', y='关键词', orientation='h',
                                    title="Top 20 高频关键词", template=TEMPLATE_THEME)
                    fig_kw.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_kw, width="stretch", config=DOWNLOAD_CONFIG)
                    export_charts["🔑 关键词分析"] = fig_kw

                st.divider()
                h1, h2 = st.columns(2)
                with h1:
                    st.markdown("#### 🔥 因素相关性热力图")
                    corr_cols = [c for c in ['月销量', '月销售额($)', '价格($)', '评分', '评分数', '上架天数'] if
                                 c in df.columns]

                    # 确保有足够的数据计算相关性
                    valid_cols = [c for c in corr_cols if df[c].nunique() > 1]

                    if len(valid_cols) > 1:
                        # 计算相关性矩阵并填充空值
                        corr_matrix = df[valid_cols].corr().fillna(0)

                        # aspect='auto' 允许热力图单元格拉伸以填充长方形容器，而不是强制正方形
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            aspect='auto',  # <--- 必须加这句！允许长方形显示
                            color_continuous_scale='RdBu_r',
                            template=TEMPLATE_THEME,
                            title="相关性矩阵"
                        )

                        st.plotly_chart(fig_corr, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts["🔥 相关性热力图"] = fig_corr
                    else:
                        st.info("数据维度不足或数值单一，无法计算相关性。")

                with h2:
                    st.markdown("#### ⚖️ 帕累托分析")
                    # 1. 数据准备
                    p_df = df.sort_values('月销售额($)', ascending=False).reset_index(drop=True)

                    if not p_df.empty:
                        # 计算累计占比
                        total_revenue = df['月销售额($)'].sum()
                        # 防止总销售额为0导致除以0错误
                        if total_revenue > 0:
                            p_df['累计占比'] = p_df['月销售额($)'].cumsum() / total_revenue * 100
                        else:
                            p_df['累计占比'] = 0

                        p_df['产品占比'] = (p_df.index + 1) / len(p_df) * 100

                        # --- 关键计算：生成你要求的两个结论 ---
                        # 1. 计算头部 20% 产品贡献的销售额占比
                        idx_20_pct = int(len(p_df) * 0.2)
                        # 边界处理：如果产品少于5个，20%可能索引为0，取第一条数据
                        idx_20_pct = max(0, min(idx_20_pct, len(p_df) - 1))
                        val_20_pct_contribution = p_df.iloc[idx_20_pct]['累计占比']

                        # 2. 计算多少产品占据了 80% 的销售额
                        # 找到第一个累计占比 >= 80 的行
                        target_row = p_df[p_df['累计占比'] >= 80]
                        if not target_row.empty:
                            val_80_pct_products = target_row.iloc[0]['产品占比']
                        else:
                            val_80_pct_products = 100  # 如果没达到80%，则为100%

                        # --- 显示结论文字 ---
                        st.info(f"""
                        **💡 帕累托法则 (80/20) 验证结论:**
                        1. 头部 **20%** 的产品贡献了市场 **{val_20_pct_contribution:.1f}%** 的销售额。
                        2. 仅需前 **{val_80_pct_products:.1f}%** 的产品即可覆盖市场 **80%** 的营收。
                        """)

                        # --- 绘图逻辑 (包含V10.0的高度/坐标轴修复) ---
                        # 添加原点 (0,0) 让曲线更美观
                        start_row = pd.DataFrame({'产品占比': [0], '累计占比': [0]})
                        p_df_plot = pd.concat([start_row, p_df], ignore_index=True)

                        fig_pareto = px.line(
                            p_df_plot,
                            x='产品占比',
                            y='累计占比',
                            title="累计销售额占比曲线",
                            template=TEMPLATE_THEME
                        )

                        # 关键修复：强制固定坐标轴范围和高度，防止导出时线条消失
                        fig_pareto.update_xaxes(range=[0, 105], title="产品数量占比 (%)")
                        fig_pareto.update_yaxes(range=[0, 105], title="累计销售额占比 (%)")
                        fig_pareto.update_layout(height=500, autosize=False)

                        # 添加辅助线
                        fig_pareto.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% 营收")
                        fig_pareto.add_vline(x=20, line_dash="dash", line_color="orange", annotation_text="20% 产品")

                        st.plotly_chart(fig_pareto, width="stretch", config=DOWNLOAD_CONFIG)
                        export_charts["⚖️ 帕累托分析"] = fig_pareto
                    else:
                        st.warning("数据不足，无法绘制帕累托图")

        # ========================================================
        # Tab 5: Keepa 深度洞察 (Ultimate Pro 版)
        # ========================================================
            with tabs[4]:
                st.header("5. Keepa 深度运营洞察")
                st.caption("基于 Keepa 全字段数据，挖掘缺货机会、竞争格局与关联流量。")

                # ==========================================
                # 🛠️ 0. 数据预处理与清洗 (集中处理)
                # ==========================================

                # 1. 全量字段映射字典 (涵盖 Keepa 导出的所有核心维度)
                # 键(Key)是程序内部使用的标准变量名，值(Value)是 Keepa Excel 表头的可能的名称列表
                target_map = {
                    # --- A. 基础信息 (Basic Info) ---
                    'asin': ['ASIN'],
                    'parent_asin': ['Parent ASIN'],
                    'title': ['Title'],
                    'brand': ['Brand'],
                    'manufacturer': ['Manufacturer'],
                    'model': ['Model'],
                    'image': ['Image'],
                    'product_group': ['Product Group'],
                    'locale': ['Locale'],  # 站点
                    'type': ['Type'],  # 类型

                    # --- B. 销量与排名 (Sales & Rank) ---
                    # 核心销量文本 ("2K+ bought in past month")
                    'sales_text': ['Bought in past month'],
                    # 核心排名
                    'rank_curr': ['Sales Rank: Current'],
                    'rank_avg_90': ['Sales Rank: 90 days avg.'],
                    'rank_ref': ['Sales Rank: Reference'],  # 排名参考类目
                    # 销售趋势 (核心机会指标)
                    'rank_drops': ['Sales Rank: Drops last 90 days'],  # 90天销量脉冲
                    'rank_drop_pct': ['Sales Rank: 90 days drop %'],  # 排名下降百分比(通常代表变好)
                    'sales_change': ['90 days change % monthly sold'],  # 销量增长率

                    # --- C. 价格体系 (Price) ---
                    # Buy Box (购物车) 价格
                    'price_curr': ['Buy Box 🚚: Current', 'Buy Box: Current'],
                    'price_avg_90': ['Buy Box 🚚: 90 days avg.'],
                    'price_avg_180': ['Buy Box 🚚: 180 days avg.'],
                    'price_drop_pct': ['Buy Box 🚚: 90 days drop %'],
                    # New (自发货/新品) 价格
                    'price_new_curr': ['New: Current'],

                    # --- D. 卖家与竞争 (Competition & BuyBox) ---
                    # 卖家数量
                    'offers_fba': ['Count of retrieved live offers: New, FBA'],
                    'offers_fbm': ['Count of retrieved live offers: New, FBM'],
                    # 购物车归属
                    'bb_seller': ['Buy Box: Buy Box Seller'],  # 当前赢得购物车的卖家
                    'bb_is_fba': ['Buy Box: Is FBA'],  # 当前是否FBA
                    'used_seller': ['Buy Box Used: Buy Box Used Seller'],
                    # 市场份额 (避坑/垄断分析关键)
                    'share_amz_90': ['Buy Box: % Amazon 90 days'],  # 亚马逊自营占比
                    'share_top_90': ['Buy Box: % Top Seller 90 days'],  # 头部卖家占比
                    'share_top_180': ['Buy Box: % Top Seller 180 days'],
                    'share_top_365': ['Buy Box: % Top Seller 365 days'],

                    # --- E. 库存与缺货 (Stock & OOS) ---
                    'stock_level': ['Buy Box 🚚: Stock'],
                    'oos_90': ['Buy Box 🚚: 90 days OOS'],  # 90天缺货率

                    # --- F. 评价体系 (Reviews) ---
                    'rating_val': ['Reviews: Rating'],
                    'rating_count': ['Reviews: Rating Count'],

                    # --- G. 物理规格 (Specs) ---
                    # 产品本身
                    'item_l': ['Item: Length (cm)'],
                    'item_w': ['Item: Width (cm)'],
                    'item_h': ['Item: Height (cm)'],
                    'item_wt': ['Item: Weight (g)'],
                    'item_vol': ['Item: Dimension (cm³)'],
                    # 包装 (FBA费计算依据)
                    'pkg_l': ['Package: Length (cm)'],
                    'pkg_w': ['Package: Width (cm)'],
                    'pkg_h': ['Package: Height (cm)'],
                    'pkg_wt': ['Package: Weight (g)'],
                    'pkg_vol': ['Package: Dimension (cm³)'],
                    'pkg_qty': ['Package: Quantity'],

                    # --- H. 变体与属性 (Variations) ---
                    'var_count': ['Variation Count'],
                    'var_asins': ['Variation ASINs'],
                    'var_attrs': ['Variation Attributes'],
                    'color': ['Color'],
                    'size': ['Size'],

                    # --- I. 时间与类目 (Time & Categories) ---
                    'listed_date': ['Listed since'],  # 上架时间
                    'track_date': ['Tracking since'],  # Keepa开始追踪时间
                    'pub_date': ['Publication Date'],
                    'release_date': ['Release Date'],
                    'cat_tree': ['Categories: Tree'],  # 类目树
                    'cat_root': ['Categories: Root'],
                    'cat_sub': ['Categories: Sub'],
                    'sub_ranks': ['Sales Rank: Subcategory Sales Ranks'],

                    # --- J. 其他/AI分析用 (Misc) ---
                    'fbt': ['Freq. Bought Together'],  # 买了又买
                    'desc': ['Description & Features: Description'],  # 产品描述
                    'feat1': ['Description & Features: Feature 1'],
                    'feat2': ['Description & Features: Feature 2'],
                    'hazmat': ['Hazardous Materials'],  # 危险品
                    'trade_in': ['Trade-In Eligible']
                }

                # 2. 智能匹配列名
                final_cols = {}
                # 获取所有列名并转小写用于匹配
                df_cols_lower = {col.lower(): col for col in df.columns}

                for key, keywords in target_map.items():
                    best_match = None
                    best_score = 0

                    for kw in keywords:
                        kw_lower = kw.lower()
                        # 查找包含关键词的列
                        matches = [col for col_low, col in df_cols_lower.items() if kw_lower in col_low]

                        for match in matches:
                            # 特殊处理：Drops 字段不能包含 %
                            if key == 'rank_drops' and '%' in match:
                                continue

                            # 简单打分：越短的匹配通常越精确，或者完全匹配
                            score = 100 if match.lower() == kw_lower else (100 - len(match) + len(kw))
                            if score > best_score:
                                best_match = match
                                best_score = score

                    if best_match:
                        final_cols[key] = best_match

                # 3. 数据清洗函数 (一次性转换数字)
                def clean_numeric_cols(dataframe, cols_map):
                    df_clean = dataframe.copy()

                    # 定义必须强制转换为数字的列 key (白名单)
                    # 只有在这个列表里的 key，我们才会去执行去百分号、转数字操作
                    numeric_keys_whitelist = [
                        # 销量与排名
                        'rank_curr', 'rank_avg_90', 'rank_drops', 'rank_drop_pct', 'sales_change',
                        'sales_est',  # 兼容旧版键名
                        # 价格
                        'price_curr', 'price_avg_90', 'price_avg_180', 'price_drop_pct', 'price_new_curr',
                        'price_avg',  # 兼容旧版键名
                        # 卖家与份额
                        'offers_fba', 'offers_fbm', 'share_amz_90', 'share_top_90', 'share_top_180', 'share_top_365',
                        # 库存
                        'stock_level', 'oos_90',
                        # 评价
                        'rating_val', 'rating_count',
                        # 规格
                        'item_l', 'item_w', 'item_h', 'item_wt', 'item_vol',
                        'pkg_l', 'pkg_w', 'pkg_h', 'pkg_wt', 'pkg_vol', 'pkg_qty',
                        # 变体
                        'var_count'
                    ]

                    for key in numeric_keys_whitelist:
                        if key in cols_map:
                            col_name = cols_map[key]
                            if col_name in df_clean.columns:
                                # 转换为字符串 -> 去掉 % , 等符号 -> 转数字
                                if df_clean[col_name].dtype == 'object':
                                    df_clean[col_name] = df_clean[col_name].astype(str).str.replace('%',
                                                                                                    '').str.replace(',',
                                                                                                                    '',
                                                                                                                    regex=False)

                                df_clean[col_name] = pd.to_numeric(df_clean[col_name], errors='coerce')

                    return df_clean

                if not final_cols:
                    st.error("⚠️ 未检测到有效的 Keepa 数据列，请检查上传的文件是否包含 Keepa 导出字段。")
                else:
                    # 执行清洗
                    df = clean_numeric_cols(df, final_cols)

                    # ==========================================
                    # 🧬 1. 父体识别模块 (缓存优化)
                    # ==========================================

                    @st.cache_data(show_spinner="正在分析父体与变体关系...")
                    def identify_parent_products(input_df):
                        """智能识别父体，带缓存"""
                        df_proc = input_df.copy()

                        # A. 优先检查明确字段
                        parent_columns = ['Parent ASIN', '父ASIN', 'Parent', '父体']
                        for col in parent_columns:
                            if col in df_proc.columns:
                                df_proc['父体ID'] = df_proc[col].fillna('独立产品')
                                return df_proc

                        # B. 智能识别算法
                        def extract_core_title(title):
                            if pd.isna(title): return "未知"
                            title_str = str(title).lower()
                            remove_words = ['pack', 'set', 'size', 'color', 'with', 'for', 'the', 'and', 'new']
                            words = [w for w in title_str.split() if w not in remove_words and len(w) > 2]
                            return ' '.join(words[:5])

                        # --- 修复：自动查找正确的标题列 ---
                        target_title_col = None
                        possible_cols = ['商品标题', 'Title', 'Product Name', '标题', 'Name']
                        for col in possible_cols:
                            if col in df_proc.columns:
                                target_title_col = col
                                break

                        if target_title_col:
                            df_proc['标题核心词'] = df_proc[target_title_col].apply(extract_core_title)
                        else:
                            # 如果完全找不到标题，用 ASIN 代替
                            df_proc['标题核心词'] = df_proc['ASIN'] if 'ASIN' in df_proc.columns else 'Unknown'
                        # ------------------------------------

                        # 使用价格分箱辅助
                        # 注意：这里也要防止 '价格' 列不存在，虽然 load_data 处理过，但 Keepa 合并表可能不同
                        price_col = '价格($)' if '价格($)' in df_proc.columns else None
                        if not price_col and 'Price' in df_proc.columns: price_col = 'Price'

                        if price_col:
                            # 确保是数字
                            df_proc[price_col] = pd.to_numeric(df_proc[price_col], errors='coerce')
                            df_proc['价格段'] = pd.cut(df_proc[price_col],
                                                       bins=[0, 10, 25, 50, 100, 1000, float('inf')],
                                                       labels=['0-10', '10-25', '25-50', '50-100', '100-500',
                                                               '500+'])
                        else:
                            df_proc['价格段'] = '未知'

                        # 初步ID
                        brand_col = '品牌' if '品牌' in df_proc.columns else 'Brand'
                        if brand_col not in df_proc.columns: df_proc[brand_col] = 'Unknown'

                        df_proc['父体ID'] = df_proc[brand_col].fillna('未知') + ' | ' + df_proc[
                            '标题核心词'] + ' | ' + \
                                            df_proc['价格段'].astype(str)

                        return df_proc

                    df_with_parents = identify_parent_products(df)

                    # 聚合父体统计数据
                    parent_agg_rules = {
                        'ASIN': 'count',
                        '月销量': 'sum',
                        '大类BSR': 'min',
                        '评分': 'mean',
                        '评分数': 'sum'
                    }
                    # 添加动态列到聚合规则
                    if 'price_avg' in final_cols: parent_agg_rules[final_cols['price_avg']] = 'mean'
                    if 'oos_90' in final_cols: parent_agg_rules[final_cols['oos_90']] = 'mean'

                    # 执行聚合 (只对存在的列)
                    valid_agg_rules = {k: v for k, v in parent_agg_rules.items() if
                                       k in df_with_parents.columns}
                    parent_stats = df_with_parents.groupby('父体ID').agg(valid_agg_rules).round(2)
                    parent_stats.rename(
                        columns={'ASIN': '变体数量', '月销量': '总月销量', '大类BSR': '最好排名'},
                        inplace=True)

                    # ==========================================
                    # 📊 2. 核心分析看板 (Tab 分页)
                    # ==========================================

                    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                        "🎯 机会挖掘 (Scoring)",
                        "📈 市场概览 (Parent)",
                        "🔍 深度透视 (Metrics)",
                        "📦 尺寸分析 (Size)"
                    ])

                    # --- Tab 1: 机会挖掘 (最核心功能前置) ---
                    with sub_tab1:
                        st.markdown("#### 🚀 智能选品机会评分 V3.0 (基于 Keepa 深度数据)")

                        # --- 评分配置面板 ---
                        with st.expander("⚙️ 评分模型配置 (Expert Mode)", expanded=False):
                            col_cfg1, col_cfg2 = st.columns(2)
                            with col_cfg1:
                                st.markdown("**⚖️ 权重因子**")
                                w_sales = st.slider("销量权重", 0, 100, 35, help="基于 'Bought in past month' 或 Drops")
                                w_profit = st.slider("利润权重", 0, 100, 20, help="基于 Buy Box 价格")
                                w_comp = st.slider("竞争权重", 0, 100, 25, help="基于评论数 & 亚马逊垄断度")
                                w_growth = st.slider("潜力权重", 0, 100, 20, help="基于缺货率 & 评分改良机会")
                            with col_cfg2:
                                st.markdown("**🎯 达标红线**")
                                target_sales = st.number_input("目标月销量", value=300, step=50)
                                max_reviews = st.number_input("最大评论数 (蓝海线)", value=250, step=50)
                                price_range = st.slider("黄金价格带 ($)", 0, 300, (18, 90))

                        # 组装配置
                        score_config = {
                            'w_sales': w_sales, 'w_profit': w_profit,
                            'w_comp': w_comp, 'w_growth': w_growth,
                            'target_sales': target_sales, 'max_reviews': max_reviews,
                            'price_min': price_range[0], 'price_max': price_range[1]
                        }

                        # --- 计算评分 ---
                        try:
                            # 传入新的 scoring_logic
                            df_scored = scoring_logic.calculate_market_score(df, final_cols, score_config)
                            df_opp = df_scored[df_scored['机会分数'] > 0].copy()
                            # 1. 计算总权重用于显示百分比
                            total_w = w_sales + w_profit + w_comp + w_growth
                            if total_w == 0: total_w = 1

                            # 2. 插入算法说明面板
                            with st.expander("📝 算法揭秘：分数是如何计算的？(点击查看逻辑)", expanded=False):
                                st.markdown(f"""
                                                       ### 🔢 综合机会分数 (0-100分) 计算逻辑
                                                       当前模型基于以下 **4个维度** 进行加权评分：

                                                       1. **📈 市场需求 (占比 {round(w_sales / total_w * 100)}%)**
                                                          * **核心指标**: `Bought in past month` (月销量) 或 `Drops`。
                                                          * **计算**: 你的目标是月销 **{target_sales}** 件。达到此数值该项得满分，销量越高分数越高。

                                                       2. **💰 利润空间 (占比 {round(w_profit / total_w * 100)}%)**
                                                          * **核心指标**: `Buy Box Price` (当前价格)。
                                                          * **计算**: 价格若在 **${price_range[0]} - ${price_range[1]}** 黄金区间内，该项得满分。偏离区间会扣分。

                                                       3. **⚔️ 竞争环境 (占比 {round(w_comp / total_w * 100)}%)**
                                                          * **核心指标**: `Review Count` (评论数) 和 `Amazon Share` (自营占比)。
                                                          * **计算**: 评论数少于 **{max_reviews}** 个得分较高。
                                                          * **⚠️ 避坑机制**: 如果 Amazon 自营占购物车时间 **>50%**，该项得分会自动**减半**；若 **>80%** 则几乎不得分。

                                                       4. **🚀 增长潜力 (占比 {round(w_growth / total_w * 100)}%)**
                                                          * **核心指标**: `OOS %` (缺货率) 和 `Rating` (星级)。
                                                          * **加分彩蛋**: 
                                                            - **缺货捡漏**: 90天缺货率 > 15% (说明供不应求)。
                                                            - **改良机会**: 销量高但评分低 (3.0-3.9分) 的产品 (说明有痛点可解决)。
                                                       """)

                                st.caption("💡 *提示：你可以通过上方的 '评分模型配置' 滑块调整这些比例。*")
                            st.success(f"🔍 已分析 {len(df)} 个 ASIN，筛选出 {len(df_opp)} 个潜力产品")

                            c1, c2 = st.columns([1.8, 1.2])
                            with c1:
                                # 准备展示列
                                cols_show = ['机会分数', '核心标签', 'ASIN', 'Brand']
                                if 'Product_Img' in df_opp.columns: cols_show.insert(0, 'Product_Img')

                                # 确保数值列存在
                                df_opp['Show_Sales'] = df_opp.get('计算后销量', 0)
                                df_opp['Show_Price'] = df_opp.get('计算后价格', 0)
                                df_opp['Show_Rating'] = df_opp.get('计算后评分', 0)
                                df_opp['Show_Amz'] = df_opp.get('亚马逊占比', 0)

                                st.dataframe(
                                    df_opp[cols_show + ['Show_Sales', 'Show_Price', 'Show_Rating', 'Show_Amz']].head(
                                        50),
                                    column_config={
                                        "Product_Img": st.column_config.ImageColumn("图片", width="small"),
                                        "机会分数": st.column_config.ProgressColumn("得分", format="%d", min_value=0,
                                                                                    max_value=100),
                                        "Show_Sales": st.column_config.NumberColumn("月销量(估)", format="%d"),
                                        "Show_Price": st.column_config.NumberColumn("价格", format="$%.2f"),
                                        "Show_Rating": st.column_config.NumberColumn("评分", format="%.1f"),
                                        "Show_Amz": st.column_config.NumberColumn("Amz占比", format="%.0f%%"),
                                    },
                                    height=600,
                                    hide_index=True,
                                    width="stretch"
                                )

                            with c2:
                                if not df_opp.empty:
                                    # 气泡图
                                    fig_opp = px.scatter(
                                        df_opp.head(100),
                                        x='Show_Price', y='Show_Sales',
                                        size='机会分数', color='机会分数',
                                        hover_name='ASIN',
                                        hover_data=['核心标签', 'Brand', 'Show_Amz'],
                                        title="💎 机会矩阵: 价格 vs 销量 (颜色=得分)",
                                        labels={'Show_Price': 'BuyBox价格', 'Show_Sales': '月销量(估)'},
                                        color_continuous_scale='RdYlGn',
                                        template=TEMPLATE_THEME
                                    )
                                    fig_opp.update_layout(yaxis_type="log")  # 对数轴
                                    # 画辅助线
                                    fig_opp.add_vline(x=price_range[0], line_dash="dot", line_color="grey")
                                    fig_opp.add_vline(x=price_range[1], line_dash="dot", line_color="grey")
                                    st.plotly_chart(fig_opp, width="stretch" )

                                    st.info("""
                                    **💡 V3.0 评分模型特性:**
                                    1. **销量优先**: 自动解析 "Bought in past month" (如 2K+ bought)。
                                    2. **避坑检测**: 若 "Buy Box: % Amazon 90 days" > 50%，分数会大幅降低 (Amz垄断)。
                                    3. **改良机会**: 销量高但评分低 (3.0-3.9) 的产品会获得高分推荐。
                                    """)

                        except Exception as e:
                            st.error(f"计算出错: {str(e)}")
                            st.write("调试信息 - 现有列名:", df.columns.tolist())

                    # --- Tab 2: 市场概览 (父体维度) ---
                    with sub_tab2:
                        st.markdown("#### 🧬 父体/变体格局分析")

                        col1, col2, col3 = st.columns(3)
                        col1.metric("识别出父体数", len(parent_stats))
                        col2.metric("单变体产品数", (parent_stats['变体数量'] == 1).sum())
                        col3.metric("平均每个父体包含", f"{parent_stats['变体数量'].mean():.1f} 个变体")

                        c1, c2 = st.columns(2)
                        with c1:
                            # 父体规模 vs 销量
                            fig_parent = px.scatter(
                                parent_stats.reset_index(),
                                x='变体数量', y='总月销量', size='总月销量',
                                color='最好排名', hover_name='父体ID',
                                log_y=True, title="父体规模(变体数) vs 市场表现",
                                template=TEMPLATE_THEME
                            )
                            st.plotly_chart(fig_parent, width="stretch")

                        with c2:
                            # 变体数量分布
                            fig_hist = px.histogram(parent_stats, x='变体数量', nbins=20,
                                                    title="变体数量分布直方图",
                                                    template=TEMPLATE_THEME)
                            st.plotly_chart(fig_hist, width="stretch")

                        with st.expander("查看完整父体数据表"):
                            st.dataframe(parent_stats, width="stretch")

                    # --- Tab 3: 深度透视 (回溯原始数据版) ---
                    with sub_tab3:
                        st.markdown("#### 🚨 缺货与补货监控")

                        # 1. 缺货监控 (保持不变)
                        col_oos = final_cols.get('oos_90')
                        col_rank = final_cols.get('rank_curr')
                        col_brand = final_cols.get('brand')

                        if col_oos and col_rank:
                            try:
                                plot_df = df.copy()
                                plot_df[col_oos] = pd.to_numeric(plot_df[col_oos], errors='coerce').fillna(0)
                                plot_df[col_rank] = pd.to_numeric(plot_df[col_rank], errors='coerce').fillna(0)
                                df_active = plot_df[(plot_df[col_oos] > 0) & (plot_df[col_rank] > 0)]

                                if not df_active.empty:
                                    fig_oos = px.scatter(
                                        df_active, x=col_rank, y=col_oos,
                                        color=col_brand if (
                                                    col_brand and col_brand in df_active.columns) else None,
                                        hover_name='ASIN', log_x=True,
                                        title=f"缺货率 vs 排名",
                                        labels={col_oos: "90天缺货率 (%)", col_rank: "当前排名 (BSR)"},
                                        template=TEMPLATE_THEME
                                    )
                                    fig_oos.add_hline(y=20, line_dash="dash", line_color="red",
                                                      annotation_text="严重缺货")
                                    st.plotly_chart(fig_oos, width="stretch")
                                else:
                                    st.info("没有检测到缺货数据 (所有产品缺货率均为 0)。")
                            except Exception as e:
                                st.warning(f"缺货图表生成失败: {e}")
                        else:
                            st.caption("未找到缺货率 (OOS) 或排名列，跳过缺货分析。")

                        st.divider()

                        # =================================================
                        # 🧹 0. 数据清洗中心 (使用 df_raw 救火)
                        # =================================================

                        # 1. 定义正则解析函数
                        def _parse_numeric_nuclear(val):
                            import re
                            # 转字符串，如果已经是NaN则返回
                            if pd.isna(val): return float('nan')
                            s = str(val).strip()
                            if not s: return float('nan')

                            # 寻找数字片段
                            match = re.search(r"(\d[\d,]*\.?\d*)", s)
                            if match:
                                clean_str = match.group(1).replace(',', '')
                                try:
                                    return float(clean_str)
                                except:
                                    return float('nan')
                            return float('nan')

                        # 2. 锁定价格列
                        target_price_col = None

                        # 优先从映射取
                        if 'price_curr' in final_cols:
                            target_price_col = final_cols['price_curr']

                        # 备用搜索
                        if not target_price_col:
                            potential = [c for c in df.columns if 'Buy Box' in c and 'Current' in c]
                            if potential: target_price_col = potential[0]

                        # 3. 执行清洗 (关键：从 df_raw 读取原始字符串！)
                        df['__Clean_Price'] = float('nan')

                        if target_price_col:
                            # 尝试从 df_raw 获取，因为 df 可能已经被转坏了
                            source_df = df_raw if 'df_raw' in locals() and target_price_col in df_raw.columns else df

                            if target_price_col in source_df.columns:
                                st.toast(
                                    f"正在从 {'原始数据(df_raw)' if source_df is df_raw else '处理数据(df)'} 中提取价格...")
                                df['__Clean_Price'] = source_df[target_price_col].apply(_parse_numeric_nuclear)
                            else:
                                st.error(f"❌ 列名 {target_price_col} 不存在于数据源中。")

                        # 4. 执行销售额清洗
                        target_rev_col = '月销售额($)'
                        df['__Clean_Revenue'] = float('nan')
                        if target_rev_col in df.columns:
                            # 销售额通常比较干净，直接用 df 即可，或者也用 df_raw
                            df['__Clean_Revenue'] = df[target_rev_col].apply(_parse_numeric_nuclear)

                        # ==========================================
                        # 🔍 调试与绘图
                        # ==========================================
                        st.markdown(f"#### 💰 价格区间效能分析")
                        if target_price_col:
                            st.caption(f"✅ 使用价格列: **{target_price_col}**")

                            # 再次检查
                            valid_count = df['__Clean_Price'].notna().sum()

                            if valid_count == 0:
                                st.error("⚠️ 依然无法提取价格！请检查下方原始数据预览。")
                                with st.expander("🔍 原始数据长什么样？", expanded=True):
                                    # 尝试显示 df_raw 的内容
                                    if 'df_raw' in locals() and target_price_col in df_raw.columns:
                                        st.write("数据源: df_raw (未被处理过的原始值)")
                                        st.dataframe(
                                            df_raw[[target_price_col]].head(10).astype(str))  # 强制转字符串显示
                                    else:
                                        st.write("数据源: df (已被处理过，可能已变成NaN)")
                                        st.dataframe(df[[target_price_col]].head(10))
                                # 停止后续运行
                                # st.stop()

                        # 1. 过滤有效数据
                        df_plot = df.dropna(subset=['__Clean_Price', '__Clean_Revenue']).copy()
                        df_plot = df_plot[(df_plot['__Clean_Price'] > 0) & (df_plot['__Clean_Revenue'] > 0)]

                        if not df_plot.empty:
                            # A. 自动去极值
                            max_price = df_plot['__Clean_Price'].quantile(0.99)
                            df_plot = df_plot[df_plot['__Clean_Price'] <= max_price]

                            # B. 切分区间
                            import math
                            if len(df_plot) > 1:
                                min_p = math.floor(df_plot['__Clean_Price'].min())
                                max_p = math.ceil(df_plot['__Clean_Price'].max())

                                step = (max_p - min_p) / 10
                                if step == 0: step = 1
                                custom_bins = [min_p + i * step for i in range(11)]

                                df_plot['Price_Bin'] = pd.cut(df_plot['__Clean_Price'], bins=custom_bins,
                                                              include_lowest=True)

                                # C. 聚合
                                stats = df_plot.groupby('Price_Bin', observed=True).agg({
                                    '__Clean_Revenue': 'mean',
                                    'ASIN': 'count'
                                })
                                stats = stats.sort_index().reset_index()

                                # D. 格式化标签
                                def format_label(interval):
                                    if pd.isna(interval): return "Unknown"
                                    left, right = interval.left, interval.right
                                    if (right - left) >= 5:
                                        return f"${int(left)}-${int(right)}"
                                    else:
                                        return f"${left:.1f}-${right:.1f}"

                                stats['Range'] = stats['Price_Bin'].apply(format_label)

                                # E. 绘图
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=stats['Range'], y=stats['__Clean_Revenue'],
                                    name='平均月销售额', marker_color='#2ECC71', yaxis='y1'
                                ))
                                fig.add_trace(go.Scatter(
                                    x=stats['Range'], y=stats['ASIN'],
                                    name='产品数量', marker_color='#E74C3C',
                                    mode='lines+markers', yaxis='y2'
                                ))

                                fig.update_layout(
                                    title=f"BuyBox 价格区间 vs 销售额",
                                    yaxis=dict(title='平均月销售额 ($)', side='left'),
                                    yaxis2=dict(title='产品数量', side='right', overlaying='y', showgrid=False),
                                    template=TEMPLATE_THEME,
                                    legend=dict(orientation="h", y=1.1),
                                    xaxis=dict(type='category', categoryorder='array',
                                               categoryarray=stats['Range'].tolist())
                                )
                                st.plotly_chart(fig, width="stretch")

                                try:
                                    best_row = stats.loc[stats['__Clean_Revenue'].idxmax()]
                                    st.success(
                                        f"📊 **分析结论**: Buy Box 价格在 **{best_row['Range']}** 的产品平均产出最高。")
                                except:
                                    pass
                            else:
                                st.warning("数据过于集中，无法切分区间。")
                        else:
                            st.warning(f"⚠️ 没有有效数据。清洗后的有效行数: {len(df_plot)}")

                        # 3. 关联流量 FBT
                        st.divider()
                        c_fbt, c_drops = st.columns(2)
                        with c_fbt:
                            st.markdown("#### 🔗 FBT 关联覆盖率")
                            if 'fbt' in final_cols:
                                fbt_ratio = (df[final_cols['fbt']].notna().sum() / len(df)) * 100
                                fig_gauge = go.Figure(go.Indicator(
                                    mode="gauge+number", value=fbt_ratio,
                                    title={'text': "拥有 FBT 数据的 ASIN 占比"},
                                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#8E44AD"}}
                                ))
                                st.plotly_chart(fig_gauge, width="stretch")
                            else:
                                st.caption("无 FBT 数据")

                        with c_drops:
                            st.markdown("#### 🔥 销售活跃度 (Drops)")
                            if 'rank_drops' in final_cols:
                                fig_drops = px.histogram(
                                    df, x=final_cols['rank_drops'], nbins=20,
                                    title="90天排名下降次数 (Drops Count)",
                                    color_discrete_sequence=['#E74C3C'],
                                    template=TEMPLATE_THEME
                                )
                                st.plotly_chart(fig_drops, width="stretch")
                            else:
                                st.caption("无 Drops 数据")
                    with sub_tab4:
                        # ==========================================
                        # 🛠️ 补丁开始：修复数据映射与清洗 (防止 NaN 报错)
                        # ==========================================

                        # 1. 修正列名映射 (让你的 c_sales 能找到数据)
                        c_l = final_cols.get('pkg_l')
                        c_w = final_cols.get('pkg_w')
                        c_h = final_cols.get('pkg_h')
                        c_price = final_cols.get('price_curr')

                        # 优先找 sales_text (新版)，找不到再找 sales_est (旧版)
                        c_sales = final_cols.get('sales_text') if 'sales_text' in final_cols else final_cols.get(
                            'sales_est')

                        # 2. 预先清洗销量数据 (防止后续转数字变成空值)
                        # 如果销量列存在，且是文本格式 (如 "2K+ bought")，我们先把它洗成数字
                        if c_sales and c_sales in df.columns:
                            def _temp_parse_sales(val):
                                import re
                                s = str(val).lower()
                                if 'k' in s:
                                    nums = re.findall(r"([\d\.]+)", s)
                                    return float(nums[0]) * 1000 if nums else 0
                                elif '+' in s or 'bought' in s:
                                    nums = re.findall(r"(\d+)", s)
                                    return float(nums[0]) if nums else 0
                                else:
                                    return pd.to_numeric(val, errors='coerce')

                            # 直接修改主 df，这样你下面的 df_spec = df[...] 就能取到干净的数字了
                            if df[c_sales].dtype == 'object':
                                df[c_sales] = df[c_sales].apply(_temp_parse_sales)

                        # ==========================================
                        # 🛠️ 补丁结束，下面是你原本的代码 (仅修改了一处 dropna)
                        # ==========================================

                        # 2. 检查数据有效性
                        if all([c_l, c_w, c_h, c_sales]):
                            st.markdown("### 📏 尺寸区间 vs 市场效益")
                            st.caption(
                                "逻辑流：先看市场主流尺寸分布，再看各尺寸段的赚钱能力（销量 & 销售额）。")

                            # --- A. 数据清洗与预处理 ---
                            # 1. 确定悬停显示的列名（如果表里没有 Parent ASIN，就自动降级显示 ASIN，防止报错）
                            hover_col = 'Parent ASIN' if 'Parent ASIN' in df.columns else 'ASIN'

                            # 2. 在截取数据时，把 hover_col 加进去
                            cols_to_keep = [c_l, c_w, c_h, c_sales, 'Title', 'Brand', 'ASIN']
                            if hover_col not in cols_to_keep:
                                cols_to_keep.append(hover_col)

                            df_spec = df[cols_to_keep].copy()

                            # 引入价格计算 GMV
                            if c_price and c_price in df.columns:
                                df_spec['Price'] = pd.to_numeric(
                                    df[c_price].astype(str).str.replace(r'[^\d.]', '', regex=True),
                                    errors='coerce')
                            else:
                                df_spec['Price'] = 0

                            # 强制转数值
                            for c in [c_l, c_w, c_h, c_sales]:
                                df_spec[c] = pd.to_numeric(df_spec[c], errors='coerce')

                            # 🔴 关键修复：除了长宽高，必须把销量的 NaN 也去掉，否则 px.scatter 的 size 参数会报错！
                            df_spec = df_spec.dropna(subset=[c_l, c_w, c_h, c_sales])

                            # 计算核心指标
                            df_spec['最长边'] = df_spec[[c_l, c_w, c_h]].max(axis=1)
                            df_spec['月销售额'] = df_spec[c_sales] * df_spec['Price']

                            # 定义尺寸区间 (Binning)
                            bins = [0, 10, 20, 30, 45, 60, 100, 999]
                            labels = ['0-10cm', '10-20cm', '20-30cm', '30-45cm', '45-60cm', '60-100cm',
                                      '100cm+']
                            df_spec['尺寸段'] = pd.cut(df_spec['最长边'], bins=bins, labels=labels)

                            # --- B. 聚合统计 ---
                            # 按尺寸段统计：产品数量、平均销量、总销售额
                            size_stats = df_spec.groupby('尺寸段', observed=True).agg({
                                'ASIN': 'count',
                                c_sales: 'mean',
                                '月销售额': 'sum'
                            }).reset_index()

                            size_stats.columns = ['尺寸区间', '产品数量', '平均月销量', '总销售额']
                            size_stats['销售额占比'] = (size_stats['总销售额'] / size_stats[
                                '总销售额'].sum()) * 100

                            # --- C. 可视化分析 ---

                            # 1. 第一层：市场存量分布 (大家都做多大的？)
                            c1, c2 = st.columns([1, 2])

                            with c1:
                                st.markdown("#### 1️⃣ 市场尺寸分布")
                                fig_pie = px.pie(
                                    size_stats,
                                    names='尺寸区间',
                                    values='产品数量',
                                    title="竞品尺寸区间占比",
                                    hole=0.4,
                                    template=TEMPLATE_THEME
                                )
                                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig_pie, width="stretch")
                                st.caption("👈 市场上最多的尺寸类型。")

                            with c2:
                                st.markdown("#### 2️⃣ 尺寸 vs 效益 (哪种卖得好？)")
                                # 双轴图：柱状图(平均销量) + 折线图(销售额占比)
                                fig_dual = go.Figure()

                                # 柱状图：平均销量
                                fig_dual.add_trace(go.Bar(
                                    x=size_stats['尺寸区间'],
                                    y=size_stats['平均月销量'],
                                    name='平均月销量 (件)',
                                    marker_color='#3498DB',
                                    yaxis='y1'
                                ))

                                # 折线图：销售额占比 (反映市场份额)
                                fig_dual.add_trace(go.Scatter(
                                    x=size_stats['尺寸区间'],
                                    y=size_stats['销售额占比'],
                                    name='销售额份额 (%)',
                                    marker_color='#E74C3C',
                                    mode='lines+markers',
                                    yaxis='y2'
                                ))

                                fig_dual.update_layout(
                                    title="各尺寸段：单品平均销量 vs 市场金额份额",
                                    yaxis=dict(title="平均月销量 (件)", side="left", showgrid=False),
                                    yaxis2=dict(title="销售额份额 (%)", side="right", overlaying="y",
                                                showgrid=False),
                                    legend=dict(x=0, y=1.1, orientation='h'),
                                    template=TEMPLATE_THEME
                                )
                                st.plotly_chart(fig_dual, width="stretch")
                                st.caption(
                                    "📊 **解读**：蓝色柱子高代表该尺寸**单品好卖**；红色折线高代表该尺寸**市场吸金能力强**。")

                            st.divider()

                            # =================================================
                            # 🟢 修正点：从这里开始，删除了重复的定义代码，只保留一套完整的逻辑
                            # =================================================
                            st.markdown("#### 3️⃣ 黄金尺寸散点分布 (长 vs 宽)")

                            # --- 1. 定义限制标准 ---
                            LIMIT_L = 91.44  # 36 inch
                            LIMIT_W = 63.50  # 25 inch
                            LIMIT_H = 63.50  # 25 inch

                            st.caption(f"💡 **图表解读**：\n"
                                       f"1. 辅助线表示限制：长 {LIMIT_L}cm | 宽 {LIMIT_W}cm。\n"
                                       f"2. **气泡颜色**：代表高度 (颜色越黄越厚)。\n"
                                       f"3. **悬停标题**：已更改为 **{hover_col}**。")

                            # --- 2. 绘制纯净版散点图 (去掉了边缘直方图) ---
                            # 智能计算坐标轴范围
                            max_l = df_spec[c_l].quantile(0.99) * 1.1
                            max_w = df_spec[c_w].quantile(0.99) * 1.1
                            max_l = max(max_l, LIMIT_L * 1.05)
                            max_w = max(max_w, LIMIT_W * 1.05)

                            fig_scatter = px.scatter(
                                df_spec,
                                x=c_l,
                                y=c_w,
                                size=c_sales,  # 销量决定大小
                                color=c_h,  # 高度决定颜色
                                hover_name=hover_col,
                                hover_data={
                                    c_l: ':.1f', c_w: ':.1f', c_h: ':.1f',
                                    c_sales: True, '月销售额': ':,.0f'
                                },
                                # 🔴 移除了 marginal_x/y，解决了"右边怪怪的"问题
                                title=f"📦 单品尺寸分布 (气泡越大数据越好 | 辅助线: {LIMIT_L} x {LIMIT_W})",
                                labels={c_l: "包装长度 (cm)", c_w: "包装宽度 (cm)", c_h: "高度",
                                        c_sales: "月销量"},
                                template=TEMPLATE_THEME,
                                opacity=0.75
                            )

                            # 样式优化
                            fig_scatter.update_layout(
                                coloraxis_colorscale='Turbo',
                                height=600,  # 适当调低高度，因为没有直方图了
                                legend=dict(orientation="h", y=1.1)
                            )

                            # 添加辅助线
                            fig_scatter.add_vline(x=LIMIT_L, line_dash="dash", line_color="red",
                                                  annotation_text="长限制")
                            fig_scatter.add_hline(y=LIMIT_W, line_dash="dash", line_color="orange",
                                                  annotation_text="宽限制")

                            # 锁定范围
                            fig_scatter.update_xaxes(range=[0, max_l])
                            fig_scatter.update_yaxes(range=[0, max_w])

                            st.plotly_chart(fig_scatter, width="stretch", config=DOWNLOAD_CONFIG)

                            # --- 3. 新增：真正直观的"销量"分布图 (Bar Chart) ---
                            st.markdown("#### 📊 哪个尺寸段产生的销量最多？(按 5cm 区间聚合)")

                            # 3.1 数据处理：将长度按 5cm 分桶
                            bin_size = 5
                            df_spec['Len_Bin'] = (df_spec[c_l] // bin_size * bin_size).astype(int)

                            # 3.2 聚合：统计每个尺寸段的总销量，而不是产品数量
                            size_sales_stats = df_spec.groupby('Len_Bin')[c_sales].sum().reset_index()
                            size_sales_stats['Label'] = size_sales_stats['Len_Bin'].astype(str) + '-' + (
                                    size_sales_stats['Len_Bin'] + bin_size).astype(str) + 'cm'

                            # 3.3 绘图：柱状图
                            fig_bar = px.bar(
                                size_sales_stats,
                                x='Label',
                                y=c_sales,
                                text=c_sales,  # 在柱子上显示具体销量
                                title="🏆 各长度区间总销量统计 (Total Sales per Size Range)",
                                labels={'Label': '长度区间 (cm)', c_sales: '该区间总月销量'},
                                template=TEMPLATE_THEME,
                                color=c_sales,
                                color_continuous_scale='Blues'
                            )

                            fig_bar.update_traces(texttemplate='%{text:.2s}',
                                                  textposition='outside')  # 销量显示简写 (如 12k)
                            fig_bar.update_layout(xaxis_title="产品包装长度区间", yaxis_title="累计月销量",
                                                  height=400)

                            # 标记出限制线所在的区间
                            fig_bar.add_vline(x=LIMIT_L / bin_size - 0.5, line_dash="dot", line_color="red",
                                              annotation_text="36in限制")

                            st.plotly_chart(fig_bar, width="stretch", config=DOWNLOAD_CONFIG)

                            # -------------------------------------------------
                            # 🔢 文字版结论
                            # -------------------------------------------------
                            # 统计有多少产品超标
                            over_l = df_spec[df_spec[c_l] > LIMIT_L].shape[0]
                            over_w = df_spec[df_spec[c_w] > LIMIT_W].shape[0]
                            over_h = df_spec[df_spec[c_h] > LIMIT_H].shape[0]
                            total_items = len(df_spec)

                            st.info(f"""
                                                                    **📏 尺寸合规统计 (限制: {LIMIT_L} x {LIMIT_W} x {LIMIT_H} cm):**
                                                                    - 🔴 **长度超标**: {over_l} 个 ({over_l / total_items:.1%})
                                                                    - 🟠 **宽度超标**: {over_w} 个 ({over_w / total_items:.1%})
                                                                    - 🔵 **高度超标**: {over_h} 个 ({over_h / total_items:.1%})

                                                                    💡 **选品建议**：请参考上方的柱状图。柱子最高的区间意味着**买家购买需求最旺盛**，而不仅仅是做的人多。
                                                                    """)

                            # --- D. 详细数据表 ---
                            st.markdown("#### 🏆 最佳尺寸标杆 (Top Sellers)")

                            top_size_items = df_spec.sort_values(c_sales, ascending=False).head(10)

                            # 修复 int64 错误
                            max_sales_val = int(
                                top_size_items[c_sales].max()) if not top_size_items.empty else 100

                            st.dataframe(
                                top_size_items[['Title', c_sales, '月销售额', c_l, c_w, c_h]],
                                column_config={
                                    "Title": st.column_config.TextColumn("标题", width="medium"),
                                    c_sales: st.column_config.ProgressColumn(
                                        "月销量", format="%d", min_value=0, max_value=max_sales_val
                                    ),
                                    "月销售额": st.column_config.NumberColumn("月销售额($)",
                                                                              format="$%.0f"),
                                    c_l: st.column_config.NumberColumn("长(cm)", format="%.1f"),
                                    c_w: st.column_config.NumberColumn("宽(cm)", format="%.1f"),
                                    c_h: st.column_config.NumberColumn("高(cm)", format="%.1f"),
                                },
                                width="stretch",
                                hide_index=True
                            )
                        else:
                            st.warning("⚠️ 数据中缺少尺寸(Package L/W/H)或销量数据，无法进行尺寸分析。")
                # ==========================================
                # 📥 数据导出
                # ==========================================
                st.divider()
                if st.button("📥 下载 Keepa 深度分析报告"):
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "点击保存 CSV",
                        data=csv,
                        file_name='keepa_deep_analysis.csv',
                        mime='text/csv'
                    )

                    # 还原：底部文字总结
            st.divider()
            st.success("📊 **分析总结 (Insights):**")
            total_sales = df['月销售额($)'].sum()
            if total_sales > 0:
                monopoly_rate = df.groupby('品牌')['月销售额($)'].sum().sort_values(ascending=False).head(
                    10).sum() / total_sales * 100
                new_product_rate = df[df['是否新品'] == '新品 (<90天)'][
                                       '月销售额($)'].sum() / total_sales * 100 if '是否新品' in df.columns else 0

                st.markdown(f"""
                  - **市场概况**: 当前样本包含 **{len(df)}** 个ASIN，总月销售额 **${total_sales:,.0f}**。
                  - **品牌垄断**: Top 10 品牌占据了市场 **{monopoly_rate:.1f}%** 的份额。
                  - **新品机会**: 过去3个月上架的新品占据了 **{new_product_rate:.1f}%** 的市场份额。
                  - **分析建议**: 结合上方的"属性分析"与"增长率矩阵"，优先开发高增长低竞争的细分属性。
                  """)
                # ==========================================
                # 🤖 AI 智能分析模块 (调用独立文件)
                # ==========================================
                st.markdown("---")
                st.subheader("🤖 AI 深度选品顾问")
                st.caption("基于当前清洗后的数据，调用大模型生成专业分析报告")

                # 配置区域
                with st.expander("⚙️ 配置 AI 模型 (DeepSeek / OpenAI / Kimi)", expanded=False):
                    c_api1, c_api2, c_api3 = st.columns(3)
                    user_api_key = c_api1.text_input("API Key", type="password", help="输入你的 API Key")
                    user_base_url = c_api2.text_input("Base URL", value="https://api.deepseek.com",
                                                      help="OpenAI 填 https://api.openai.com/v1")
                    user_model = c_api3.text_input("Model Name", value="deepseek-chat",
                                                   help="例如 gpt-4o, deepseek-chat")

                # 触发按钮
                if st.button("✨ 生成 AI 深度报告", type="primary"):
                    if not user_api_key:
                        st.warning("⚠️ 请先在上方配置 API Key")
                    else:
                        # 创建一个空容器用于流式输出
                        report_box = st.empty()
                        full_text = ""

                        with st.spinner("🤖 AI 正在分析数据并撰写报告..."):
                            # --- 调用独立模块 ---
                            stream_response = ai_analysis.get_market_analysis_stream(
                                df=df,
                                api_key=user_api_key,
                                base_url=user_base_url,
                                model_name=user_model,
                                target_attr=target_attr
                            )

                            # 处理返回结果
                            if isinstance(stream_response, str) and stream_response.startswith("Error"):
                                st.error(f"调用失败: {stream_response}")
                            else:
                                # 流式渲染
                                for chunk in stream_response:
                                    if chunk.choices[0].delta.content is not None:
                                        full_text += chunk.choices[0].delta.content
                                        # 实时更新 UI，加上光标效果
                                        report_box.markdown(full_text + "▌")

                                # 渲染完成，移除光标
                                report_box.markdown(full_text)
                                st.success("✅ 分析完成")

                                # (可选) 如果你想把 AI 结论也放入导出图表字典中，可以在这里操作
                                # export_charts["AI_Report"] = full_text

            # --- 导出逻辑 (V9.0: 原版 HTML 结构 + JSON图表注入) ---
            if st.sidebar.button("🔄 生成交互式HTML报告"):
                with st.spinner("正在生成完整报告..."):
                    # 准备分析数据
                    analysis_data = {
                        'total_products': len(df),
                        'avg_monthly_sales': df['月销量'].mean(),
                        'avg_monthly_revenue': df['月销售额($)'].mean(),
                        'avg_growth_rate': avg_growth_val,  # 使用修正后的增长率
                        'top_brands': df.groupby('品牌')['月销售额($)'].sum().sort_values(ascending=False).head(
                            5),
                        'total_revenue': total_sales
                    }

                    # 生成
                    html_content = generate_interactive_html_report(df, export_charts, analysis_data,
                                                                    target_attr)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    filename = f"Amazon_Analysis_Report_{timestamp}.html"

                    # 下载链接
                    st.sidebar.markdown(create_download_link(html_content, filename, "📥 下载修复版交互报告"),
                                        unsafe_allow_html=True)
                    st.sidebar.success("✅ 报告生成成功！(增长率计算已修正)")

if __name__ == "__main__":
    main()
