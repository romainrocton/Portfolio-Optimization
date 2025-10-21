from typing import Final, Sequence
import pandas as pd
import os

# --- Colors for Streamlit (CSS) ---
THEME_COLORS: Final[dict[str, str]] = {
    "primary": "#6366F1",
    "accent": "#8B5CF6",
    "badge_bg": "rgba(99,102,241,0.18)",
    "section_bg": "rgba(255,255,255,0.6)",  # semi-transparent background
    "border": "rgba(148,163,184,0.35)",
    "metric_bg": "rgba(99,102,241,0.18)",
    "text_primary": "inherit",
    "text_secondary": "inherit",
    "tag_bg": "#EF4444",
    "tag_text": "#FFFFFF",
    "slider_active": "#6366F1",
    "slider_thumb": "#EF4444",
}

# Dictionary mapping indices to their countries

dico_pays={}
dico_pays["CAC 40"]="France"    
dico_pays["DAX 40"]="Germany"
dico_pays["AEX 25"]="Netherlands"
dico_pays["IBEX 35"]="Spain"
dico_pays["FTSE MIB"]="Italy"

# Dictionary mapping stocks to their sectors
file_path = os.path.join("sectors", "Sectors.xlsx")
df_sectors = pd.read_excel(file_path)
nb_total_assets=df_sectors.shape[0]
dico_sectors={}
for i in range(nb_total_assets):
    if df_sectors.iloc[i,2] not in dico_sectors:
        dico_sectors[df_sectors.iloc[i,2]]=[df_sectors.iloc[i,1]]
    else:
        dico_sectors[df_sectors.iloc[i,2]].append((df_sectors.iloc[i,1]))