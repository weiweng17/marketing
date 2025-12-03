# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re


def parse_bought_count(val):
    """解析 'Bought in past month' 字段，例如 '2K+ bought' -> 2000"""
    if pd.isna(val): return 0
    s = str(val).lower()
    if 'k' in s:
        num = re.findall(r"([\d\.]+)", s)
        return float(num[0]) * 1000 if num else 0
    elif '+' in s or 'bought' in s:
        num = re.findall(r"(\d+)", s)
        return float(num[0]) if num else 0
    else:
        try:
            return float(s)
        except:
            return 0


def calculate_market_score(df, col_map, config):
    """
    Keepa 深度数据评分模型 V3.0
    """
    data = df.copy()

    # 1. 映射关键列 (基于你提供的 Keepa 标准字段)
    c_sales_text = col_map.get('sales_text', 'Bought in past month')  # 文本型销量
    c_drops = col_map.get('rank_drops', 'Sales Rank: Drops last 90 days')
    c_price = col_map.get('price_curr', 'Buy Box 🚚: Current')  # 优先用 BuyBox 价格
    c_reviews = col_map.get('rating_count', 'Reviews: Rating Count')
    c_rating = col_map.get('rating_val', 'Reviews: Rating')
    c_oos = col_map.get('oos_90', 'Buy Box 🚚: 90 days OOS')
    c_amazon = col_map.get('amazon_share', 'Buy Box: % Amazon 90 days')  # 亚马逊自营占比
    c_fba_offers = col_map.get('offers_fba', 'Count of retrieved live offers: New, FBA')
    c_fbm_offers = col_map.get('offers_fbm', 'Count of retrieved live offers: New, FBM')

    scores = []

    # 2. 获取权重配置
    w_sales = config.get('w_sales', 30)
    w_profit = config.get('w_profit', 20)
    w_comp = config.get('w_comp', 25)
    w_growth = config.get('w_growth', 25)

    # 阈值配置
    target_sales = config.get('target_sales', 300)
    max_reviews = config.get('max_reviews', 200)
    price_min = config.get('price_min', 15)
    price_max = config.get('price_max', 80)

    # 3. 预处理数据 (加速计算)
    # 解析文本销量: "2K+ bought" -> 2000
    if c_sales_text in data.columns:
        data['__calc_sales'] = data[c_sales_text].apply(parse_bought_count)
    else:
        data['__calc_sales'] = 0

    # 4. 逐行评分
    for idx, row in data.iterrows():
        reasons = []

        # --- A. 销量分 (Demand) ---
        # 逻辑：优先看 "Bought in past month"，如果没有，看 "Drops"
        sales_val = row.get('__calc_sales', 0)
        drops_val = pd.to_numeric(row.get(c_drops, 0), errors='coerce')
        if pd.isna(drops_val): drops_val = 0

        s_sales = 0
        if sales_val > 0:
            # 这里的 target_sales 通常是月销300
            s_sales = min(100, (sales_val / target_sales) * 100)
            if sales_val > 1000: reasons.append("🔥月销1k+")
        elif drops_val > 0:
            # 如果没有具体销量，用 Drops 估算。一般 30个drops 约等于月销30-50 (不准确但可用)
            s_sales = min(100, (drops_val / 30) * 80)  # Drops 权重稍微低一点
            if drops_val > 60: reasons.append("📉高频出单")

        # --- B. 利润分 (Profit) ---
        price = pd.to_numeric(row.get(c_price, 0), errors='coerce')
        if pd.isna(price): price = 0

        s_profit = 0
        if price_min <= price <= price_max:
            s_profit = 100
        elif price > 0:
            # 偏离惩罚
            dist = min(abs(price - price_min), abs(price - price_max))
            s_profit = max(0, 100 - dist * 3)
            if price < 10: reasons.append("⚠️低价")

        # --- C. 竞争分 (Competition) ---
        s_comp = 0

        # C1. 评论数
        reviews = pd.to_numeric(row.get(c_reviews, 0), errors='coerce')
        if pd.isna(reviews): reviews = 9999

        rev_score = 0
        if reviews < max_reviews:
            rev_score = 100 - (reviews / max_reviews * 50)  # 即使接近200也有50分
            if reviews < 50: reasons.append("✨新星")

        # C2. 亚马逊自营垄断 (关键!)
        amz_share = str(row.get(c_amazon, '0')).replace('%', '').strip()
        try:
            amz_share = float(amz_share)
        except:
            amz_share = 0

        amz_penalty = 1.0
        if amz_share > 50:
            amz_penalty = 0.5  # 亚马逊占一半，分数减半
            reasons.append("🦖AMZ垄断")
        if amz_share > 80:
            amz_penalty = 0.1  # 亚马逊霸屏，几乎不得分

        # C3. 卖家数量 (FBA + FBM)
        offers = pd.to_numeric(row.get(c_fba_offers, 0), errors='coerce') + \
                 pd.to_numeric(row.get(c_fbm_offers, 0), errors='coerce')
        if offers > 0 and offers < 5:
            reasons.append("👥卖家少")

        # 综合竞争分 = 评论分 * 亚马逊惩罚系数
        s_comp = rev_score * amz_penalty

        # --- D. 潜力分 (Growth/Signal) ---
        s_growth = 0

        # D1. 缺货捡漏
        oos = pd.to_numeric(row.get(c_oos, 0), errors='coerce')
        if pd.isna(oos): oos = 0
        if oos > 15:
            s_growth += 40
            reasons.append(f"🚨缺货{int(oos)}%")

        # D2. 痛点改良 (Private Label)
        rating = pd.to_numeric(row.get(c_rating, 0), errors='coerce')
        if pd.isna(rating): rating = 0

        # 销量还可以 (超过目标的一半) 且 评分不好 (3.0 - 3.9)
        if (sales_val > target_sales * 0.5 or drops_val > 45) and 3.0 <= rating <= 3.9:
            s_growth += 60
            reasons.append("🛠️改良机会")

        s_growth = min(100, s_growth)

        # --- E. 汇总 ---
        total_w = w_sales + w_profit + w_comp + w_growth
        if total_w == 0: total_w = 1

        final_score = (
                              s_sales * w_sales +
                              s_profit * w_profit +
                              s_comp * w_comp +
                              s_growth * w_growth
                      ) / total_w

        # 补全核心数据用于展示
        scores.append({
            'index': idx,
            '机会分数': int(final_score),
            '核心标签': ' '.join(reasons[:3]),
            '计算后销量': int(sales_val) if sales_val > 0 else 0,  # 用于UI展示
            '计算后价格': price,
            '计算后评分': rating,
            '亚马逊占比': amz_share
        })

    score_df = pd.DataFrame(scores).set_index('index')
    result = pd.concat([data, score_df], axis=1)

    return result.sort_values('机会分数', ascending=False)
