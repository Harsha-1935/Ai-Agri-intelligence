import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Professional Terminal & Visual Settings
warnings.simplefilter(action='ignore')
pd.options.mode.chained_assignment = None 
sns.set_theme(style="white", palette="muted")

# ============================================================
# 1. DATA INTEGRATION & UNIT NORMALIZATION
# ============================================================
print("--- Phase 1: Data Integration & Unit Normalization ---")

def clean_name(x):
    return "".join(filter(str.isalnum, str(x).upper()))

try:
    s1, s2 = pd.read_csv('375 SOIL NUTRIETS.csv'), pd.read_csv('375 SOIL NUTRIETS2.csv')
    df_soil = pd.concat([s1, s2], ignore_index=True)
    df_soil.columns = df_soil.columns.str.lower().str.strip().str.replace(" ", "_")
    df_soil['state'] = df_soil['state'].apply(clean_name)
    
    rain_list = []
    files = ['India_Rainfall_2014_Districts.csv', 'India_Rainfall_Monthly_Districts_2016.csv',
             'India_Rainfall_Monthly_Districts_2017.csv', 'India_Rainfall_Monthly_Districts_2020.csv',
             'India_Rainfall_Monthly_Districts_2021.csv']
             
    for f in files:
        try:
            tmp = pd.read_csv(f, encoding='latin1')
            tmp.columns = tmp.columns.str.lower().str.strip()
            subset = tmp[['state', 'jun', 'jul', 'aug', 'sep']].copy()
            subset[['jun', 'jul', 'aug', 'sep']] *= 10 # cm to mm
            rain_list.append(subset)
        except: pass
        
    state_rain = pd.concat(rain_list).groupby('state').mean().reset_index()
    state_rain['state'] = state_rain['state'].apply(clean_name)
    df_master = pd.merge(df_soil, state_rain, on='state', how='inner')
    df_master['total_rain'] = df_master[['jun', 'jul', 'aug', 'sep']].sum(axis=1)
    print("✅ Data Integrated Successfully (mm).")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================
# 2. AI MODEL SETUP
# ============================================================
n_col, p_col, k_col = 'n_low', 'p_low', 'k_low'
n_threshold = df_master[n_col].median()
rain_threshold = 500.0 

df_master['risk'] = df_master.apply(lambda r: 1 if r[n_col] > n_threshold or r['total_rain'] < rain_threshold else 0, axis=1)
features = ['jun', 'jul', 'aug', 'sep', 'total_rain', n_col, p_col, k_col]
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(df_master[features], df_master['risk'])

# ============================================================
# 3. YIELD LOGIC
# ============================================================
def calculate_yield(n, p, k, rain, ai_risk):
    s_score = 50 * (1 - ((n + p + k) / 300))
    if 800 <= rain <= 1500: r_score = 50
    elif rain < 800: r_score = 50 * (rain / 800)
    elif 1500 < rain <= 2500: r_score = 50 * ((3500 - rain) / 2000)
    elif 2500 < rain <= 3500: r_score = 10
    else: r_score = -100
    total = s_score + r_score
    if ai_risk == 1 and total > 35: total = 35.0
    return 0.0 if (rain > 3600 or total < 0) else max(0, min(100, total))

# ============================================================
# 4. DASHBOARDS (4-PLOT & 3-PLOT HYBRID LAYOUT)
# ============================================================
print("📊 Generating Dashboards with Customized Spacing...")

# --- DASHBOARD 1: 4 Plots (2x2) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 11))
fig1.suptitle('ENVIRONMENTAL RISK ANALYSIS DASHBOARD', fontsize=18, fontweight='bold', y=0.97)
sns.countplot(ax=axes1[0,0], x='risk', data=df_master, hue='risk', palette='coolwarm', legend=False)
axes1[0,0].set_title("1. Risk Distribution", pad=12)
sns.scatterplot(ax=axes1[0,1], x=n_col, y='total_rain', hue='risk', data=df_master, palette='RdYlGn_r', s=80)
axes1[0,1].set_title("2. Nitrogen Def vs. Rainfall", pad=12)
sns.heatmap(df_master[features].corr(), ax=axes1[1,0], annot=True, cmap='Spectral', fmt=".2f", linewidths=0.5)
axes1[1,0].set_title("3. Feature Correlation Matrix", pad=12)
sns.kdeplot(ax=axes1[1,1], data=df_master, x='total_rain', hue='risk', fill=True, palette='viridis', alpha=0.5)
axes1[1,1].set_title("4. Rainfall Density Curve", pad=12)
plt.subplots_adjust(hspace=0.5, top=0.88, wspace=0.3, bottom=0.08)

# --- DASHBOARD 2: 4 Plots (2x2) ---
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 11))
fig2.suptitle('SEASONAL MONSOON PERFORMANCE DASHBOARD', fontsize=18, fontweight='bold', y=0.97)
months = ['jun', 'jul', 'aug', 'sep']
for i, m in enumerate(months):
    r, c = i // 2, i % 2
    sns.boxplot(ax=axes2[r,c], x='risk', y=m, data=df_master, hue='risk', palette='Pastel1', legend=False)
    axes2[r,c].set_title(f"{i+5}. {m.upper()} Rainfall Spread (mm)", pad=12)
plt.subplots_adjust(hspace=0.5, top=0.88, wspace=0.3, bottom=0.08)

# --- DASHBOARD 3: 3 Plots Side-by-Side (1x3) ---
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 7))
fig3.suptitle('SOIL NUTRIENT & VULNERABILITY INSIGHTS', fontsize=18, fontweight='bold', y=0.94)
df_master[[n_col, p_col, k_col]].plot(kind='box', ax=axes3[0], color='teal', patch_artist=True)
axes3[0].set_title("9. NPK Deficiency Spread", pad=15)
sns.regplot(ax=axes3[1], x=p_col, y=k_col, data=df_master, scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
axes3[1].set_title("10. Phosphorus vs Potassium Link", pad=15)
sns.violinplot(ax=axes3[2], x='risk', y=n_col, data=df_master, hue='risk', split=True, palette='Set2', legend=False)
axes3[2].set_title("11. Nitrogen Vulnerability Profile", pad=15)
plt.subplots_adjust(top=0.82, wspace=0.3, bottom=0.15, left=0.05, right=0.95)

plt.show()

# ============================================================
# 5. PREDICTOR SYSTEM (WITH OUTLOOK LOGIC)
# ============================================================
print("\n" + "="*50 + "\n🌾 AGRO-INTELLIGENCE: PREDICTION SYSTEM\n" + "="*50)
try:
    j = float(input("Enter June Rain (mm): "))
    jl = float(input("Enter July Rain (mm): "))
    a = float(input("Enter August Rain (mm): "))
    s = float(input("Enter September Rain (mm): "))
    n = float(input("Enter Nitrogen Def %: "))
    p = float(input("Enter Phosphorus Def %: "))
    k = float(input("Enter Potassium Def %: "))
    
    tr = j + jl + a + s
    user_df = pd.DataFrame([[j, jl, a, s, tr, n, p, k]], columns=features)
    risk_class = model.predict(user_df)[0]
    yield_pot = calculate_yield(n, p, k, tr, risk_class)
    
    print("\n" + "★"*50)
    print(f"📊 AI STATUS: {'🚩 HIGH RISK' if risk_class == 1 else '✅ STABLE'}")
    print(f"📈 YIELD POTENTIAL: {yield_pot:.1f}%")
    
    # --- ADDED OUTLOOK LOGIC ---
    if tr > 3600:
        print("🚫 OUTLOOK: TOTAL CROP FAILURE (Severe Flood > 3600mm).")
    elif risk_class == 1:
        print("🚩 OUTLOOK: CRITICAL. Environmental stress is too high for success.")
    elif yield_pot < 45:
        print("⚠️ OUTLOOK: POOR. Significant nutrient/water management required.")
    elif yield_pot < 75:
        print("🚜 OUTLOOK: AVERAGE. Harvest is stable but needs optimization.")
    else:
        print("🌟 OUTLOOK: GOOD TO EXCELLENT.")
    print("★"*50 + "\n")
except:
    print("❌ Invalid Input.")
