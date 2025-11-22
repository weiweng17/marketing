# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from openai import OpenAI


def get_market_analysis_stream(df, api_key, base_url, model_name, target_attr=None):
    """
    独立模块 V2.0：深度多维数据投喂，生成专家级报告。
    """
    try:
        # ==========================================
        # 1. 数据深度挖掘 (Data Mining)
        # ==========================================

        # --- A. 基础大盘 ---
        total_rev = df['月销售额($)'].sum()
        avg_price = df['价格($)'].mean()
        total_sales = df['月销量'].sum()
        asin_count = len(df)

        # --- B. 竞争与垄断 (CR5) ---
        top_brands = df.groupby('品牌')['月销售额($)'].sum().sort_values(ascending=False).head(5)
        top5_share = (top_brands.sum() / total_rev * 100) if total_rev > 0 else 0
        brands_str = ", ".join([f"{b}({v / total_rev * 100:.1f}%)" for b, v in top_brands.items()])

        # --- C. 市场成熟度与痛点 (基于评分) ---
        avg_rating = df['评分'].mean() if '评分' in df.columns else 0
        avg_review_count = df['评分数'].mean() if '评分数' in df.columns else 0
        # 计算低分率 (评分低于3.8的产品占比)
        low_rating_ratio = len(df[df['评分'] < 3.8]) / len(df) * 100 if '评分' in df.columns else 0

        # --- D. 新品活力 (Barrier to Entry) ---
        # 假设 '是否新品' 列已在主程序清洗好，如果没有则尝试计算
        if '是否新品' not in df.columns and '上架天数' in df.columns:
            df['是否新品'] = df['上架天数'].apply(lambda x: '新品' if x <= 90 else '老品')

        new_product_data = "无上架时间数据"
        if '是否新品' in df.columns:
            new_products = df[df['是否新品'].str.contains('新品')]
            new_share = (new_products['月销售额($)'].sum() / total_rev * 100) if total_rev > 0 else 0
            new_count = len(new_products)
            new_product_data = f"新品(90天内)占比 {new_share:.1f}% (共{new_count}个)，新品平均营收 ${new_products['月销售额($)'].mean():,.0f}"

        # --- E. 价格带分布 (Price Segmentation) ---
        # 简单将价格分为：低端(<25%)、中端(25-75%)、高端(>75%)
        p25, p75 = df['价格($)'].quantile([0.25, 0.75])
        low_end = df[df['价格($)'] <= p25]['月销量'].sum()
        mid_end = df[(df['价格($)'] > p25) & (df['价格($)'] <= p75)]['月销量'].sum()
        high_end = df[df['价格($)'] > p75]['月销量'].sum()
        price_structure = f"低价区(<${p25:.0f})销量占比 {low_end / total_sales * 100:.0f}%, 中端区销量占比 {mid_end / total_sales * 100:.0f}%, 高端区(>${p75:.0f})销量占比 {high_end / total_sales * 100:.0f}%"

        # --- F. 属性偏好 (如果有) ---
        attr_context = "用户未选择特定属性"
        if target_attr and target_attr in df.columns:
            top_attrs = df.groupby(target_attr)['月销售额($)'].sum().sort_values(ascending=False).head(3)
            attr_list = ", ".join([f"{k}" for k in top_attrs.index])
            attr_context = f"分析属性 [{target_attr}]，最吸金的 Top3 变体为: {attr_list}"

        # ==========================================
        # 2. 构建专家级 Prompt
        # ==========================================
        system_prompt = """
        你是一位拥有 20 年经验的亚马逊首席选品官 (Chief Product Officer)。
        你的风格：逻辑严密、数据驱动、直击痛点、商业嗅觉敏锐。
        你不仅会读数据，还能通过数据推导出用户画像和潜在的商业风险。
        请不要堆砌辞藻，用最干练的语言给出建议。
        """

        user_prompt = f"""
        请基于以下全维度市场数据，为我输出一份《深度选品可行性报告》：

        【1. 市场大盘】
        - 总月收: ${total_rev:,.0f} | 样本数: {asin_count}
        - 平均客单价: ${avg_price:.2f}
        - 价格带销量结构: {price_structure}

        【2. 竞争壁垒】
        - 品牌垄断度 (CR5): {top5_share:.1f}% (Top5品牌: {brands_str})
        - 评论门槛: 平均评论数 {avg_review_count:.0f} 个
        - 新品机会: {new_product_data}

        【3. 产品质量与属性】
        - 平均评分: {avg_rating:.2f} 分
        - 差评爆雷率 (<3.8分占比): {low_rating_ratio:.1f}% (如果此值高，说明有巨大改良机会)
        - {attr_context}

        ------------------------------------------
        请按照以下 Markdown 结构输出分析（必须包含 Emoji）：

        ### 🎯 1. 核心结论 (Go / No-Go)
        用一句话判定：这是“蓝海捡钱”、“红海厮杀”还是“小而美”的市场？给出 0-10 的推荐分。

        ### 👤 2. 用户画像与痛点推演
        基于价格带和评分数据，推测买家是什么人？他们最在意什么？(如果评分低，推测他们在这个品类经常抱怨什么？)

        ### 💰 3. 黄金切入点 (Actionable Advice)
        - **定价策略**：结合价格带结构，建议新卖家定什么价位切入最容易出单？
        - **差异化方向**：如果垄断度高，建议避开什么？如果评分低，建议改良什么？
        - **属性建议**：{target_attr if target_attr else '规格'} 应该怎么选？

        ### ⚠️ 4. 死亡陷阱预警
        基于数据（如新品存活率低、巨头垄断、差评率高等），指出最可能导致亏损的因素。
        """

        # 3. 发起请求
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,  # 稍微降低温度，让分析更理性
            stream=True
        )

        return response

    except Exception as e:
        return f"Error: AI 分析模块运行出错 - {str(e)}"
