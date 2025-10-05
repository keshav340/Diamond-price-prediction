import pandas as pd
#Load and Merge Sheets

file_path = "sample-data (3).xlsx"
sheet1 = pd.read_excel(file_path, sheet_name="IN-OUT")
sheet2 = pd.read_excel(file_path, sheet_name="PRICE")

# Merge both sheets on diamond ID
data = pd.merge(sheet1, sheet2, on="ereport_no", how="left")

# Rename columns for consistency
data.rename(columns={
    data.columns[0]: "diamond_id",   # first column
    "cut_group": "cut",
    "size": "size_range",
    "disc": "discount"
}, inplace=True)


#  Basic Cleaning

data["is_sold"] = data["out_date"].notna().astype(int)
data["out_date"] = data["out_date"].fillna("Unsold")


#  Mapping Quality Attributes

color_map = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7, "K": 8, "L": 9, "M": 10}
clarity_map = {
    "FL": 1, "IF": 2, "VVS1": 3, "VVS2": 4, "VS1": 5, "VS2": 6,
    "SI1": 7, "SI2": 8, "SI3": 9, "I1": 10, "I2": 11, "I3": 12
}
cut_map = {"3EX": 1, "EX": 2, "VG": 3, "VGDN": 4, "GD": 5}
florecent_map = {"None": 1, "Faint": 2, "Medium": 3, "Strong": 4}

# Clean and map fluorescence column
data["florecent"] = (
    data["florecent"].fillna("None")
    .str.strip()
    .str.capitalize()
    .map(florecent_map)
)

# Map categorical columns
data["color"] = data["color"].map(color_map)
data["clarity"] = data["clarity"].map(clarity_map)
data["cut"] = data["cut"].map(cut_map)


# 4. Feature Engineering

def range_to_mid(value):
    try:
        low, high = map(float, value.split("-"))
        return (low + high) / 2
    except:
        return None

data["size_mid"] = data["size_range"].apply(range_to_mid)

# Remove outliers based on discount (IQR method)
Q1, Q3 = data["discount"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = data.query("@Q1 - 1.5 * @IQR <= discount <= @Q3 + 1.5 * @IQR")

# Sort by in_date
df_clean = df_clean.sort_values(by="in_date")

# Category grouping
group_cols = ["shape", "size_range", "color", "clarity", "cut", "florecent"]
df_clean["category_count"] = df_clean.groupby(group_cols)["discount"].transform("count")

#  Quality Scoring System

color_score = ((11 - data["color"]) / 10) * 100
clarity_score = ((13 - data["clarity"]) / 12) * 100
cut_score = ((6 - data["cut"]) / 5) * 100
florecent_score = ((5 - data["florecent"]) / 4) * 100

df_clean["quality_score"] = (
    0.40 * cut_score +
    0.30 * color_score +
    0.20 * clarity_score +
    0.10 * florecent_score
)

# Lag Features
df_clean["discount_lag1"] = df_clean.groupby("diamond_id")["discount"].shift(1)
df_clean["discount_lag2"] = df_clean.groupby("diamond_id")["discount"].shift(2)
df_clean["discount_diff"] = df_clean["discount"] - df_clean["discount_lag1"]

df_clean[["discount_lag1", "discount_lag2", "discount_diff"]] = (
    df_clean[["discount_lag1", "discount_lag2", "discount_diff"]].fillna(0)
)

df_clean["in_date"] = pd.to_datetime(df_clean["in_date"], errors="coerce")
df_clean["out_date"] = pd.to_datetime(df_clean["out_date"], errors="coerce")
df_clean["tenure_days"] = (df_clean["out_date"] - df_clean["in_date"]).dt.days
df_clean["tenure_days"] = df_clean["tenure_days"].fillna(0)

def categorize_tenure(days):
    if days <= 15:
        return "Short-term"
    elif days <= 30:
        return "Medium-term"
    else:
        return "Long-term"

df_clean["tenure_category"] = df_clean["tenure_days"].apply(categorize_tenure)

print(df_clean.isnull().sum())

df_clean.to_csv("preprocessed_diamond.csv", index=False)
print("Preprocessed diamond data saved successfully.")
