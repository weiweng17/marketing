import pandas as pd
import numpy as np
import io


def process_and_merge_amazon_tables(df_main, df_keepa):
    """
    功能：
    1. 清洗两个表格的 ASIN 格式。
    2. 检查 Keepa 表的 'Parent ASIN'，如果是空值，自动用 'ASIN' 填充。
    3. 将两个表格合并为一个超级大表。
    4. 返回合并后的 DataFrame。

    参数:
    df_main: 表1 DataFrame (包含运营/销售数据)
    df_keepa: 表2 DataFrame (包含 Keepa 历史数据)
    """

    # --- 步骤 1: 基础清洗 (去除 ASIN 空格，防止匹配失败) ---
    # 确保列名存在，防止报错
    if 'ASIN' in df_main.columns:
        df_main['ASIN'] = df_main['ASIN'].astype(str).str.strip()

    if 'ASIN' in df_keepa.columns:
        df_keepa['ASIN'] = df_keepa['ASIN'].astype(str).str.strip()
    else:
        raise ValueError("错误：表2 (Keepa) 中找不到 'ASIN' 列，无法进行合并。")

    # --- 步骤 2: 核心逻辑 - 填充 Parent ASIN ---
    # 检查是否存在 Parent ASIN 列
    target_col = 'Parent ASIN'
    if target_col in df_keepa.columns:
        # 将空字符串、纯空格转换为 NaN (真正的空值)，方便处理
        df_keepa[target_col] = df_keepa[target_col].replace(r'^\s*$', np.nan, regex=True)

        # 核心功能：如果 Parent ASIN 为空，则填入 ASIN 的值
        df_keepa[target_col] = df_keepa[target_col].fillna(df_keepa['ASIN'])
    else:
        print(f"提示：表2中未发现 '{target_col}' 列，跳过自动填充步骤。")

    # --- 步骤 3: 合并表格 ---
    # 使用左连接 (Left Join)，保留表1的所有数据
    # suffixes 参数用于处理重名字段，会自动加上后缀区分
    merged_df = pd.merge(
        df_main,
        df_keepa,
        on='ASIN',
        how='left',
        suffixes=('_Main', '_Keepa')
    )

    return merged_df


def convert_df_to_excel_bytes(df):
    """
    功能：将 DataFrame 转换为内存中的 Excel 文件流 (用于网页下载，不生成本地文件)
    """
    output = io.BytesIO()
    # 使用 xlsxwriter 引擎，通常比默认引擎兼容性更好
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Super_Merged_Data')

    # 指针回到开始位置
    output.seek(0)
    return output
