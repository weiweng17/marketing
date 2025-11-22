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
import numpy as np

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

        # 2. 表头清洗
        df.columns = df.columns.str.strip()

        # 3. 货币与数字清洗
        cols_to_clean = ['月销售额($)', '价格($)', 'FBA($)', '子体销售额($)', '买家运费($)']
        for col in cols_to_clean:
            if col in df.columns:
                # 增加处理 'Free' 或 '-' 等非数字字符
                df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[^\d.-]', '', x))
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 4. 百分比清洗 (还原：不做除法，保留原值)
        percent_cols = ['毛利率', '留评率', '月销量增长率']
        for col in percent_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                # 严格遵守原版逻辑：不自动除以100

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

            # 新品逻辑
            df['是否新品'] = df['上架天数'].apply(lambda x: '新品 (<90天)' if x <= 90 else '老品')

        # 7. 属性列标准化
        attr_cols = ['品牌', '大类目', '配送方式', 'BuyBox类型', '商品标题']
        for col in attr_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', 'Unknown').replace('', 'Unknown')
            elif col in ['品牌', '大类目']:
                df[col] = 'Unknown'

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

# 替换整个 generate_interactive_html_report 函数
def generate_interactive_html_report(df, charts_data, analysis_data, target_attr=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 准备图表 HTML
    charts_html = ""
    for chart_name, fig in charts_data.items():
        # --- 核心修复 1: 强制冻结尺寸 ---
        # 不要让浏览器去猜大小，直接写死像素，保证导出后和看到的一模一样
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
        # 这样文件会变大(3MB+)，但绝对能显示。这里我建议你先试 True
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
# 3. 主程序
# ==========================================
def main():
    # 侧边栏
    st.sidebar.title("🛠️ 分析控制台")
    uploaded_file = st.sidebar.file_uploader("上传市场调研数据 (Excel/CSV)", type=['xlsx', 'csv'])

    # 还原：底部签名
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Data Analysis Tool | 阿伟出品")

    if uploaded_file:
        df_raw = load_data(uploaded_file)

        if df_raw is not None:
            df = df_raw.copy()

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
            valid_attrs = [c for c in all_cols if c.startswith('参数_')] + \
                          [c for c in all_cols if
                           any(x in c for x in ['颜色', 'Color', '材质', 'Material', '尺寸', 'Size'])]
            target_attr = st.sidebar.selectbox("🎯 选择重点分析属性", valid_attrs) if valid_attrs else None

            # --- 计算核心指标 (严格遵守 V3.4 逻辑) ---
            # 还原：增长率乘以100
            avg_growth_val = df['月销量增长率'].mean() * 100 if '月销量增长率' in df.columns else 0

            # --- 主界面 ---
            st.title("🚀 亚马逊全维度市场扫描报告")

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

                    st.plotly_chart(fig_matrix, use_container_width=True, config=DOWNLOAD_CONFIG)
                    export_charts["📈 产品潜力四象限分析"] = fig_matrix

            with c2:
                # 还原：增长率排行榜 DataFrame
                st.markdown("#### 📈 所有ASIN增长率排行榜")
                if '月销量增长率' in df.columns:
                    growth_ranking = df[['ASIN', '月销量', '月销量增长率', '价格($)', '品牌']].copy()
                    growth_ranking['月销量增长率'] = growth_ranking['月销量增长率'] * 100
                    growth_ranking = growth_ranking.sort_values('月销量增长率', ascending=False).head(20)

                    st.dataframe(
                        growth_ranking,
                        hide_index=True,
                        column_config={
                            "ASIN": st.column_config.TextColumn("ASIN", width="small"),
                            "月销量": st.column_config.ProgressColumn("月销量", format="%d", min_value=0,
                                                                      max_value=int(df['月销量'].max())),
                            "月销量增长率": st.column_config.NumberColumn("增长率 (%)", format="%.1f%%"),
                            "价格($)": st.column_config.NumberColumn("价格", format="$%.2f")
                        },
                        height=500
                    )

            st.divider()

            # Tab 页结构
            tabs = st.tabs(["🧬 属性深度分析", "🏆 品牌与时间", "📦 卖家与新品", "🗝️ NLP与高级统计"])

            # Tab 1: 属性 (还原逻辑)
            with tabs[0]:
                if target_attr:
                    st.header(f"2. 属性深度分析: {target_attr}")

                    # 还原：聚合逻辑
                    df_analysis = df.copy()
                    df_analysis['月销量增长率_显示'] = df_analysis['月销量增长率'] * 100

                    attr_group = df_analysis.groupby(target_attr).agg({
                        '月销量': 'sum',
                        '月销售额($)': 'sum',
                        '月销量增长率_显示': 'mean',
                        '价格($)': 'mean',
                        'ASIN': 'count'
                    }).reset_index()

                    top_attrs = attr_group.sort_values('月销售额($)', ascending=False).head(15)
                    top_vals = top_attrs[target_attr].tolist()

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
                            legend=dict(orientation="h", y=1.1)
                        )
                        st.plotly_chart(fig_combo, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts[f"💰 {target_attr} 销售分析"] = fig_combo

                    with t2:
                        # 价格分布
                        filtered_attr_df = df[df[target_attr].isin(top_vals)]
                        fig_box = px.box(filtered_attr_df, x=target_attr, y="价格($)", color=target_attr,
                                         title=f"{target_attr} 价格分布", template=TEMPLATE_THEME)
                        fig_box.update_layout(showlegend=False)
                        st.plotly_chart(fig_box, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts[f"💰 {target_attr} 价格分析"] = fig_box

                    # 增长率 Bar
                    fig_growth = px.bar(top_attrs, x=target_attr, y="月销量增长率_显示", color="月销量增长率_显示",
                                        color_continuous_scale="RdYlGn", title=f"🚀 {target_attr} 增长趋势",
                                        template=TEMPLATE_THEME)
                    fig_growth.update_yaxes(title_text="月销量增长率 (%)")
                    fig_growth.update_traces(hovertemplate='%{x}<br>增长率: %{y:.1f}%')
                    st.plotly_chart(fig_growth, use_container_width=True, config=DOWNLOAD_CONFIG)
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
                    st.plotly_chart(fig_pie, use_container_width=True, config=DOWNLOAD_CONFIG)
                    export_charts["🏆 品牌市场占有率"] = fig_pie

                with b2:
                    st.markdown("#### 📅 爆款通常在几月上架？")
                    if '上架月份' in df.columns:
                        high_sales_df = df[df['月销量'] > df['月销量'].mean()]
                        month_counts = high_sales_df['上架月份'].value_counts().reset_index()
                        month_counts.columns = ['月份', '数量']
                        fig_month = px.bar(month_counts, x='月份', y='数量', title="热销品上架月份分布",
                                           template=TEMPLATE_THEME)
                        st.plotly_chart(fig_month, use_container_width=True, config=DOWNLOAD_CONFIG)
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
                    st.plotly_chart(fig_time, use_container_width=True, config=DOWNLOAD_CONFIG)
                    export_charts["📅 上架时间分析"] = fig_time

            # Tab 3: 卖家 (还原逻辑)
            with tabs[2]:
                col_last1, col_last2, col_last3 = st.columns(3)
                with col_last1:
                    if '配送方式' in df.columns:
                        fig_fba = px.pie(df, names='配送方式', title='配送方式占比',
                                         color_discrete_sequence=px.colors.qualitative.Set2, template=TEMPLATE_THEME)
                        st.plotly_chart(fig_fba, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts["👥 配送方式占比"] = fig_fba
                with col_last2:
                    if 'BuyBox类型' in df.columns:
                        fig_bb = px.pie(df, names='BuyBox类型', title='卖家类型占比',
                                        color_discrete_sequence=px.colors.qualitative.Set3, template=TEMPLATE_THEME)
                        st.plotly_chart(fig_bb, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts["👥 卖家类型占比"] = fig_bb
                with col_last3:
                    if '是否新品' in df.columns:
                        new_share = df.groupby('是否新品')['月销售额($)'].sum().reset_index()
                        fig_new = px.pie(new_share, values='月销售额($)', names='是否新品', title='新品市场占有率',
                                         color='是否新品',
                                         color_discrete_map={'新品 (<90天)': '#2ECC71', '老品': '#95A5A6'},
                                         template=TEMPLATE_THEME)
                        st.plotly_chart(fig_new, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts["👥 新品市场占有率"] = fig_new

            # Tab 4: 高级 (保留新增功能，但修复导出)
            with tabs[3]:
                st.markdown("#### 🗝️ NLP 标题高频词")
                kw_df = analyze_keywords(df)
                if kw_df is not None:
                    fig_kw = px.bar(kw_df.head(20), x='出现频次', y='关键词', orientation='h',
                                    title="Top 20 高频关键词", template=TEMPLATE_THEME)
                    fig_kw.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_kw, use_container_width=True, config=DOWNLOAD_CONFIG)
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

                        st.plotly_chart(fig_corr, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts["🔥 相关性热力图"] = fig_corr
                    else:
                        st.info("数据维度不足或数值单一，无法计算相关性。")

                with h2:
                    st.markdown("#### ⚖️ 帕累托分析")
                    p_df = df.sort_values('月销售额($)', ascending=False).reset_index(drop=True)
                    p_df['累计占比'] = p_df['月销售额($)'].cumsum() / p_df['月销售额($)'].sum() * 100
                    p_df['产品占比'] = (p_df.index + 1) / len(p_df) * 100

                    st.caption("💡 结论: 头部产品贡献了绝大部分销售额")

                    # 1. 确保数据非空
                    if not p_df.empty:
                        # 2. 添加起始点 (0,0)，确保线条从原点出发（优化视觉）
                        start_row = pd.DataFrame({'产品占比': [0], '累计占比': [0]})
                        p_df = pd.concat([start_row, p_df], ignore_index=True)

                        fig_pareto = px.line(p_df, x='产品占比', y='累计占比', title="80/20法则分析",
                                             template=TEMPLATE_THEME,
                                             render_mode='svg')  # <--- 必须加这句！强制用 SVG 画线

                        # 强制加粗线条，设置显眼的颜色
                        fig_pareto.update_traces(line=dict(color='#E74C3C', width=4))

                        # 设置坐标轴范围，稍微留点空隙
                        fig_pareto.update_xaxes(title="产品数量占比 (%)", range=[-1, 101])
                        fig_pareto.update_yaxes(title="累计销售额占比 (%)", range=[-1, 105])

                        fig_pareto.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% 营收")
                        fig_pareto.add_vline(x=20, line_dash="dash", line_color="orange", annotation_text="20% 产品")

                        st.plotly_chart(fig_pareto, use_container_width=True, config=DOWNLOAD_CONFIG)
                        export_charts["⚖️ 帕累托分析"] = fig_pareto
                    else:
                        st.warning("数据不足，无法绘制帕累托图")

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

            # --- 导出逻辑 (V9.0: 原版 HTML 结构 + JSON图表注入) ---
            if st.sidebar.button("🔄 生成交互式HTML报告"):
                with st.spinner("正在生成完整报告..."):
                    # 准备分析数据
                    analysis_data = {
                        'total_products': len(df),
                        'avg_monthly_sales': df['月销量'].mean(),
                        'avg_monthly_revenue': df['月销售额($)'].mean(),
                        'avg_growth_rate': avg_growth_val,  # 使用修正后的增长率
                        'top_brands': df.groupby('品牌')['月销售额($)'].sum().sort_values(ascending=False).head(5),
                        'total_revenue': total_sales
                    }

                    # 生成
                    html_content = generate_interactive_html_report(df, export_charts, analysis_data, target_attr)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    filename = f"Amazon_Analysis_Report_{timestamp}.html"

                    # 下载链接
                    st.sidebar.markdown(create_download_link(html_content, filename, "📥 下载修复版交互报告"),
                                        unsafe_allow_html=True)
                    st.sidebar.success("✅ 报告生成成功！(增长率计算已修正)")


if __name__ == "__main__":
    main()
