"""
Learnagement Dashboard - Configuration et Traitement des Données
Dashboard basé sur la base de données learnagement.sql
Structure native APC (Approche Par Compétences)

"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
import numpy as np
from datetime import datetime
import networkx as nx
from io import StringIO
import sqlite3
import json
import dash_leaflet as dl

# ═══════════════════════════════════════════════════════════════════
#                        CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder='assets',
    title="Learnagement - Tableau de bord pédagogique",
    external_stylesheets=['https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap']
)

# ═══════════════════════════════════════════════════════════════════
#                        PARTIE RESEAU POLYTECH (Section 1)
# ═══════════════════════════════════════════════════════════════════

def reparer_texte(texte):
    if not isinstance(texte, str):
        return texte
    try:
        return texte.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            return texte.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texte

def count_competences(competences):
    if not isinstance(competences, list):
        return 0
    if len(competences) > 0 and isinstance(competences[0], str) and "Erreur" in competences[0]:
        return 0
    return len(competences)

def count_metiers(metiers):
    if not isinstance(metiers, list):
        return 0
    flat_metiers = []
    for item in metiers:
        if isinstance(item, list):
            flat_metiers.extend(item)
        elif isinstance(item, str):
            flat_metiers.append(item)
    return len([m for m in flat_metiers if isinstance(m, str) and len(m) > 3])

file_name = 'data.json'
try:
    with open(file_name, 'r', encoding='utf-8') as f:
        datavf = json.load(f)
except FileNotFoundError:
    print(f"Erreur: Le fichier {file_name} est introuvable.")
    datavf = {}

coords_ecoles = {
    "Polytech Nantes": [47.282, -1.520], "Polytech Montpellier": [43.632, 3.863],
    "Polytech Annecy": [45.920, 6.138], "Polytech Paris saclay": [48.706, 2.169],
    "Polytech Tours": [47.354, 0.704], "Polytech Nice Sophia": [43.616, 7.072],
    "Polytech Angers": [47.481, -0.594], "Polytech Clermont": [45.758, 3.111],
    "Polytech Grenoble": [45.193, 5.767], "Polytech Lyon": [45.783, 4.868],
    "Polytech Nancy": [48.665, 6.155]
}

records = []
for school, formations in datavf.items():
    for f_data in formations:
        raw_nom = f_data.get('formation', 'Inconnu')
        nom_clean = reparer_texte(raw_nom.replace('-', ' ').title())
        records.append({
            'Ecole': school,
            'Formation': nom_clean,
            'Nombre de Compétences': count_competences(f_data.get('competences', [])),
            'Nombre de Métiers': count_metiers(f_data.get('metiers', [])),
            'Secteurs': f_data.get('secteurs', []),
            'Metiers_Bruts': f_data.get('metiers', [])
        })

df_polytech = pd.DataFrame(records)
df_filtered = df_polytech[df_polytech['Nombre de Compétences'] > 0].copy()

# ═══════════════════════════════════════════════════════════════════
#                    CHARGEMENT DES DONNÉES SQL
# ═══════════════════════════════════════════════════════════════════

def load_learnagement_data(sql_file_path):
    """
    Charge les données depuis learnagement.sql et crée les DataFrames principaux
    Structure native de la base de données APC
    """
    import re

    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sql_content = f.read()

    def parse_insert_values(table_name, sql_text):
        pattern = rf"INSERT INTO `{table_name}`.*?VALUES\s+(.*?);"
        match = re.search(pattern, sql_text, re.DOTALL | re.IGNORECASE)

        if not match:
            return []

        values_text = match.group(1).replace('NULL', 'None')
        rows = []

        for line in values_text.split('\n'):
            line = line.strip()
            if line.startswith('(') and (line.endswith('),') or line.endswith(')')):
                line = line.rstrip(',').rstrip(';')
                try:
                    rows.append(eval(line))
                except:
                    pass

        return rows

    competences_data = parse_insert_values('APC_competence', sql_content)
    df_competences = pd.DataFrame(
        competences_data,
        columns=['id_competence', 'libelle_competence', 'code_competence', 'description']
    )

    niveaux_data = parse_insert_values('APC_niveau', sql_content)
    df_niveaux = pd.DataFrame(
        niveaux_data,
        columns=['id_niveau', 'id_competence', 'niveau', 'libelle_niveau']
    )

    ac_data = parse_insert_values('APC_apprentissage_critique', sql_content)
    df_apprentissages = pd.DataFrame(
        ac_data,
        columns=['id_apprentissage_critique', 'id_niveau', 'libelle_apprentissage']
    )

    ac_modules_data = parse_insert_values('APC_apprentissage_critique_as_module', sql_content)
    df_ac_modules = pd.DataFrame(
        ac_modules_data,
        columns=['id_apprentissage_critique', 'id_module', 'type_lien']
    )

    modules_data = parse_insert_values('MAQUETTE_module', sql_content)
    df_modules = pd.DataFrame(
        modules_data,
        columns=['id_module', 'code_module', 'nom', 'ECTS', 'id_discipline', 'id_semestre',
                 'hCM', 'hTD', 'hTP', 'hTPTD', 'hPROJ', 'hPersonnelle', 'id_responsable', 'commentaire']
    )

    composantes_data = parse_insert_values('APC_composante_essentielle', sql_content)
    df_composantes = pd.DataFrame(
        composantes_data,
        columns=['id_composante_essentielle', 'id_competence', 'libelle_composante_essentielle']
    )

    #  Métiers / situations professionnelles
    situations_data = parse_insert_values('APC_situation_professionnelle', sql_content)
    df_situations = pd.DataFrame(
        situations_data,
        columns=['id_situation_professionnelle', 'id_competence', 'libelle_situation']
    )

    return df_competences, df_niveaux, df_apprentissages, df_ac_modules, df_modules, df_composantes, df_situations


df_competences, df_niveaux, df_apprentissages, df_ac_modules, df_modules, df_composantes, df_situations = load_learnagement_data(
    "learnagement.sql"
)

# ═══════════════════════════════════════════════════════════════════
#                    TRAITEMENT DES DONNÉES - VERSION UNIFIÉE
# ═══════════════════════════════════════════════════════════════════

df_base = (
    df_apprentissages
    .merge(df_niveaux, on='id_niveau', how='left')
    .merge(df_competences, on='id_competence', how='left')
)

ac_modules = (
    df_ac_modules
    .groupby('id_apprentissage_critique')
    .agg({
        'id_module': lambda x: list(x),
        'type_lien': lambda x: list(x)
    })
    .reset_index()
)
ac_modules.columns = ['id_apprentissage_critique', 'modules_list', 'types_lien_list']

df_main = df_base.merge(ac_modules, on='id_apprentissage_critique', how='left')

df_main['modules_list'] = df_main['modules_list'].apply(lambda x: x if isinstance(x, list) else [])
df_main['types_lien_list'] = df_main['types_lien_list'].apply(lambda x: x if isinstance(x, list) else [])

df_main['nb_modules'] = df_main['modules_list'].apply(len)
df_main['has_module'] = df_main['nb_modules'] > 0

df_main['id_module'] = df_main['modules_list'].apply(lambda x: x[0] if len(x) > 0 else 0)
df_main['type_lien'] = df_main['types_lien_list'].apply(lambda x: x[0] if len(x) > 0 else 'Non associé')

df_main['competence_label'] = df_main['code_competence']

competence_labels = {
    'COMP_IDU1': 'COMP_IDU1',
    'COMP_IDU2': 'COMP_IDU2',
    'COMP_IDU3': 'COMP_IDU3',
    'COMP_IDU4': 'COMP_IDU4'
}

df_main['niveau_code'] = df_main['competence_label'] + '-N' + df_main['niveau'].astype(str)

module_names = {}
for _, row in df_modules.iterrows():
    module_id = row['id_module']
    code = row['code_module'] if pd.notna(row['code_module']) else f"M{module_id}"
    nom = row['nom'] if pd.notna(row['nom']) else "Module"
    module_names[module_id] = f"{code} - {nom}"

module_names[0] = "Non associé"
module_names[0.0] = "Non associé"

module_ids = {name: id_mod for id_mod, name in module_names.items()}

semestre_map = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6]
}
df_main['semestres'] = df_main['niveau'].map(lambda x: semestre_map.get(x, []))

df_main['lien_requis'] = df_main['type_lien'] == 'Requis'

mask_sans_module = df_main['id_module'].isna()
nb_ac_sans_module = mask_sans_module.sum()
if nb_ac_sans_module > 0:
    df_main.loc[mask_sans_module, 'id_module'] = 0.0
    df_main.loc[mask_sans_module, 'type_lien'] = 'Non associé'

# ═══════════════════════════════════════════════════════════════════
#  ORIENTATION PAR MÉTIER 
# ═══════════════════════════════════════════════════════════════════

df_situations["libelle_situation"] = (
    df_situations["libelle_situation"]
    .astype(str)
    .apply(reparer_texte)
    .str.strip()
)
metiers_sql = sorted(df_situations["libelle_situation"].dropna().unique().tolist())


df_ac_sem = (
    df_ac_modules.merge(
        df_modules[["id_module", "id_semestre"]],
        on="id_module",
        how="left"
    )
    .merge(
        df_apprentissages[["id_apprentissage_critique", "id_niveau", "libelle_apprentissage"]],
        on="id_apprentissage_critique",
        how="left"
    )
    .merge(
        df_niveaux[["id_niveau", "id_competence", "niveau"]],
        on="id_niveau",
        how="left"
    )
    .merge(
        df_competences[["id_competence", "code_competence"]],
        on="id_competence",
        how="left"
    )
)

df_ac_sem["id_semestre"] = pd.to_numeric(df_ac_sem["id_semestre"], errors="coerce")
df_ac_sem["code_competence"] = df_ac_sem["code_competence"].astype(str)


df_ac_sem = df_ac_sem[(df_ac_sem["id_semestre"] >= 1) & (df_ac_sem["id_semestre"] <= 10)]

# ═══════════════════════════════════════════════════════════════════
#                      DESIGN SYSTEM COULEURS
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "primary": "#3B82F6",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#06B6D4",
    "light": "#F8FAFC",
    "dark": "#1E293B",

    "competences": {
        "COMP_IDU1": "#FF513F",
        "COMP_IDU2": "#FFA03F",
        "COMP_IDU3": "#3B82F6",
        "COMP_IDU4": "#10B981",
    },

    "niveaux": {
        1: "#BFDBFE",
        2: "#60A5FA",
        3: "#2563EB",
    },

    "type_lien": {
        "Requis": "#EF4444",
        "Recommandé": "#F59E0B",
        "Complémentaire": "#10B981",
    },
}

def get_competence_color(competence_label):
    return COLORS["competences"].get(competence_label, "#777777")

def get_niveau_color(niveau):
    return COLORS["niveaux"].get(niveau, "#AAAAAA")

# ═══════════════════════════════════════════════════════════════════
#                    FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def parse_competencies(row):
    comps = []
    if 'niveau_code' in row.index and pd.notna(row['niveau_code']):
        comps.append(row['niveau_code'])
    return comps

def compute_competency_counts_per_module(df_input):
    df_work = df_input.copy()
    if "libelle_apprentissage" not in df_work.columns:
        df_work = df_work.assign(libelle_apprentissage=df_work.index.astype(str))

    df_work = df_work.assign(All_Competencies=df_work["niveau_code"])
    df_exploded = df_work.copy()

    df_pivot_global = (
        df_exploded
        .groupby(["niveau", "competence_label"], as_index=False)
        .size()
        .pivot(index="niveau", columns="competence_label", values="size")
        .fillna(0)
        .astype("Float64")
    )

    df_count = (
        df_exploded
        .groupby(["niveau", "id_module", "competence_label"], as_index=False)
        .size()
    )

    df_pivot_niveau = (
        df_count
        .pivot(index=["niveau", "id_module"], columns="competence_label", values="size")
        .fillna(0)
        .astype("Float64")
    )

    df_pivot_module = (
        df_exploded
        .pivot(
            index=["niveau", "id_module", "libelle_apprentissage"],
            columns="competence_label",
            values="type_lien"
        )
        .fillna("")
    )

    return df_pivot_global, df_pivot_niveau, df_pivot_module


def create_heatmap_for_global(df_pivot_global):
    fig = go.Figure()
    for j in range(df_pivot_global.shape[1]):
        comp = df_pivot_global.columns[j]
        color_comp = get_competence_color(comp)
        color_scale = ["white", color_comp]

        df_comp = df_pivot_global.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA

        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_pivot_global.columns,
                y=df_pivot_global.index,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                hovertemplate="<b>Niveau %{y}</b><br>%{x}<br>Apprentissages critiques: %{z}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Répartition des apprentissages critiques<br><sub>Cliquez sur un niveau pour voir les modules</sub>",
            x=0.5, xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis_title="Niveaux",
        yaxis=dict(type="category", autorange="reversed", showgrid=False, tickfont=dict(size=12)),
        height=600,
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=100, r=60, t=120, b=150),
        clickmode="event+select",
    )
    return fig


def create_heatmap_for_niveau(niveau, df_pivot_niveau, ylegend=True):
    df_used = df_pivot_niveau.xs(niveau, level="niveau")
    fig = go.Figure()
    z_min = 0
    z_max = df_pivot_niveau.max().max()

    for j in range(df_used.shape[1]):
        comp = df_used.columns[j]
        color_comp = get_competence_color(comp)
        color_scale = ["white", color_comp]

        df_comp = df_used.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA

        module_names_list = [module_names.get(mod, f"Module {mod}") for mod in df_used.index]

        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=module_names_list,
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1, ygap=1,
                hovertemplate="<b>%{y}</b><br>%{x}<br>AC: %{z}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Modules du niveau {niveau}<br><sub>Cliquez sur un module pour voir les apprentissages critiques</sub>",
            x=0.5, xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        yaxis_title="Modules",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        height=max(500, len(df_used.index) * 35 + 180),
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor=COLORS["light"],
        margin=dict(l=280, r=60, t=120, b=180),
        clickmode="event+select",
    )

    if not ylegend:
        fig.update_yaxes(showticklabels=False)

    return fig


def create_heatmap_for_module(module_id, df_pivot_module):
    df_reorganized = df_pivot_module.replace({
        "Requis": 3,
        "Recommandé": 2,
        "Complémentaire": 1,
        "": 0
    })

    df_used = (
        df_reorganized
        .xs(module_id, level="id_module")
        .astype("Int64")
    )

    fig = go.Figure()
    z_min = 0
    z_max = 3

    for j in range(df_used.shape[1]):
        comp = df_used.columns[j]
        color_comp = get_competence_color(comp)
        color_scale = ["white", color_comp]

        df_comp = df_used.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA

        hover_text = []
        for idx in df_used.index:
            if isinstance(idx, tuple):
                ac_text = idx[-1]
            else:
                ac_text = idx
            hover_text.append(ac_text)

        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=hover_text,
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1, ygap=1,
                customdata=df_comp.values,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Type: %{customdata}<extra></extra>",
            )
        )

    module_name = module_names.get(module_id, f"Module {int(module_id)}")
    fig.update_layout(
        title=dict(
            text=f"{module_name} - Apprentissages critiques",
            x=0.5, xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        yaxis_title="Apprentissages critiques",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10)),
        height=max(600, len(df_used.index) * 25 + 180),
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor=COLORS["light"],
        margin=dict(l=400, r=60, t=120, b=180),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#                       COMPOSANTS UI
# ═══════════════════════════════════════════════════════════════════

def create_stat_card(title, value, subtitle, color, trend=None, comparison=None):
    content = [
        html.Div([html.Div(title, className="stat-label")], className="stat-header"),
        html.Div(str(value), className="stat-value", style={"color": color}),
    ]
    if comparison:
        content.append(html.Div(comparison, className="stat-description"))
    else:
        content.append(html.Div(subtitle, className="stat-description"))
    return html.Div(content, className="card stat-card card-content")


def create_section_header(title, subtitle):
    return html.Div(
        html.Div(
            [
                html.H2(title, className="section-title"),
                html.P(subtitle, className="section-subtitle"),
            ],
            className="section-header-content",
        ),
        className="section-header",
    )


def create_filter_section():
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Filtres", className="sidebar-title"),
                    html.P("Personnalisez votre vue", className="sidebar-subtitle"),
                ],
                className="sidebar-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Niveaux", className="filter-label"),
                            dcc.Dropdown(
                                id="semester-filter",
                                options=[{"label": f"Niveau {n}", "value": n}
                                         for n in sorted(df_main["niveau"].dropna().unique())],
                                value=sorted(df_main["niveau"].dropna().unique()),
                                multi=True,
                                className="filter-input",
                                placeholder="Tous les niveaux",
                            ),
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Label("Compétences", className="filter-label"),
                            dcc.Dropdown(
                                id="ue-filter",
                                options=[{"label": comp, "value": comp}
                                         for comp in sorted(df_main["competence_label"].dropna().unique())],
                                value=sorted(df_main["competence_label"].dropna().unique()),
                                multi=True,
                                className="filter-input",
                                placeholder="Toutes les compétences",
                            ),
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Label("Types de lien", className="filter-label"),
                            dcc.Checklist(
                                id="level-filter",
                                options=[
                                    {"label": "Requis", "value": "Requis"},
                                    {"label": "Recommandé", "value": "Recommandé"},
                                    {"label": "Complémentaire", "value": "Complémentaire"},
                                    {"label": "Non associé", "value": "Non associé"},
                                ],
                                value=["Requis", "Recommandé", "Complémentaire", "Non associé"],
                            )
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Button("↻ Réinitialiser", id="reset-filters", n_clicks=0, className="btn btn-reset"),
                        ],
                        className="filter-footer",
                    ),
                    html.Div(id="filter-status", className="filter-status"),
                ],
                className="sidebar-content",
            ),
        ],
        className="sidebar-panel",
    )

def create_competence_legend_card():
    competences_info = [
        {"code": "COMP_IDU1", "label": "Concevoir et mettre en œuvre des systèmes informatiques", "color": "#FF513F"},
        {"code": "COMP_IDU2", "label": "Collecter et traiter des données numériques", "color": "#FFA03F"},
        {"code": "COMP_IDU3", "label": "Gérer les usages des données numériques en lien avec le client", "color": "#3B82F6"},
        {"code": "COMP_IDU4", "label": "Gérer un projet informatique", "color": "#10B981"},
    ]
    legend_items = []
    for comp in competences_info:
        legend_items.append(
            html.Div([
                html.Div(style={
                    "width": "20px", "height": "20px",
                    "backgroundColor": comp["color"], "borderRadius": "4px",
                    "marginRight": "12px", "flexShrink": "0"
                }),
                html.Div([
                    html.Span(comp["code"], style={"fontWeight": "600", "marginRight": "8px"}),
                    html.Span(comp["label"], style={"color": "#6B7280", "fontSize": "14px"}),
                ]),
            ], style={
                "display": "flex", "alignItems": "center",
                "marginBottom": "12px", "padding": "8px",
                "backgroundColor": "#F9FAFB", "borderRadius": "6px",
            })
        )
    return html.Div([
        html.Div([html.H3("Légende des Compétences", style={
            "margin": "0", "fontSize": "18px", "fontWeight": "600", "color": "#1F2937"
        })], style={"marginBottom": "20px"}),
        html.Div(legend_items),
    ], style={
        "backgroundColor": "white", "padding": "24px", "borderRadius": "12px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "marginBottom": "30px",
    })

def viz7_statistics_dashboard(filtered_df):
    if filtered_df.empty:
        return html.Div("Aucune donnée", style={"textAlign": "center", "padding": "20px"})

    total_ac = filtered_df["id_apprentissage_critique"].nunique()
    total_modules = sum(filtered_df['nb_modules']) if 'nb_modules' in filtered_df.columns else filtered_df['id_module'].nunique()
    total_competences = filtered_df['competence_label'].nunique()
    competences_actives = filtered_df['id_competence'].unique()
    composantes_filtrees = df_composantes[df_composantes['id_competence'].isin(competences_actives)]
    total_composantes = len(composantes_filtrees)
    nb_comp_avec_composantes = composantes_filtrees['id_competence'].nunique()

    return html.Div(
        [
            html.Div(
                [
                    create_stat_card("Apprentissages Critiques", total_ac, f"{total_modules} modules", COLORS["primary"]),
                    create_stat_card("Compétences", total_competences, "Compétences distinctes", COLORS["success"]),
                    create_stat_card("Modules", total_modules, "Modules uniques", COLORS["warning"]),
                    create_stat_card("Composantes Essentielles", total_composantes,
                                     f"Réparties sur {nb_comp_avec_composantes} compétence(s)", COLORS["danger"]),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                    "gap": "20px",
                    "marginBottom": "30px"
                }
            ),
            create_competence_legend_card(),
        ]
    )

def viz2_workload_radar(filtered_df):
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée à afficher. Veuillez ajuster vos filtres.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#94a3b8", family="Inter, sans-serif"),
        )
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=650)
        return fig

    niveaux = sorted(filtered_df["niveau"].dropna().unique())
    workload_data = []

    for niv in niveaux:
        niv_data = filtered_df[filtered_df["niveau"] == niv]
        unique_comps = niv_data["competence_label"].nunique()
        total_ac = len(niv_data)
        modules_count = niv_data["id_module"].nunique()
        
        # Compter les composantes essentielles pour les compétences de ce niveau
        competences_du_niveau = niv_data["id_competence"].unique()
        composantes_count = df_composantes[df_composantes["id_competence"].isin(competences_du_niveau)].shape[0]

        workload_data.append({
            "Niveau": f"N{niv}",
            "Compétences": unique_comps,
            "AC": total_ac,
            "Modules": modules_count,
            "Composantes": composantes_count,
        })

    wl_df = pd.DataFrame(workload_data)
    fig = go.Figure()
    for metric in ["Compétences", "AC", "Modules", "Composantes"]:
        fig.add_trace(
            go.Scatterpolar(
                r=wl_df[metric],
                theta=wl_df["Niveau"],
                fill="toself",
                name=metric,
                hovertemplate=f"{metric}: %{{r}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Couverture de travail par niveau",
        polar=dict(radialaxis=dict(visible=True)),
        height=650,
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# GRAPHIQUES Analyses principales + listes au clic
# ═══════════════════════════════════════════════════════════════════

def bar_h_comp(df_, x, y, title):
    fig = px.bar(
        df_,
        x=x,
        y=y,
        orientation="h",
        text=x,
        title=title,
        color=y,  
        color_discrete_map=COLORS["competences"],  
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def bar_h(df_, x, y, title):
    fig = px.bar(df_, x=x, y=y, orientation="h", text=x, title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=420, paper_bgcolor="white", plot_bgcolor="white")
    return fig

def empty_fig(msg="Aucune donnée"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
    return fig


# ═══════════════════════════════════════════════════════════════════
#                          LAYOUT COMPLET
# ═══════════════════════════════════════════════════════════════════

app.layout = html.Div(
    [
        dcc.Store(id="filtered-data-store"),
        dcc.Store(id="drill-down-level", data="global"),
        dcc.Store(id="selected-semester", data=None),
        dcc.Store(id="selected-module", data=None),

        # HEADER
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("L", className="header-logo-icon"),
                                html.Div(
                                    [
                                        html.H1("Learnagement - Tableau de bord", className="header-title"),
                                        html.P("Visualisez où, quand et comment développer vos compétences",
                                               className="header-subtitle"),
                                    ]
                                ),
                            ],
                            className="header-logo",
                        ),
                    ],
                    className="header-content",
                )
            ],
            className="app-header",
            style={
                "position": "sticky",
                "top": "0",
                "zIndex": "9999",
                "backgroundColor": "white",
                "width": "100%"
            }
        ),

        # ═══════════════════════════════════════════════════════════════
        # Partie polytech réseau 
        # ═══════════════════════════════════════════════════════════════
        html.Div(style={'padding': '20px', 'fontFamily': '"Poppins", sans-serif'}, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'}, children=[
                html.Div(style={'flex': '1.5'}, children=[
                    html.H3("Localisation des écoles du Reseau Polytech "),
                    dl.Map([dl.TileLayer(), dl.LayerGroup(id="layer-markers")], center=[46.5, 2.5], zoom=6,
                           style={'height': '65vh', 'borderRadius': '12px'})
                ]),
                html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'height': '65vh'}, children=[
                    html.H3(id='titre-ecole', children="🏫 Sélectionnez une école",
                            style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.Div(style={'flex': '1', 'backgroundColor': 'white', 'borderRadius': '15px',
                                   'boxShadow': '0 4px 15px rgba(0,0,0,0.05)', 'padding': '15px',
                                   'overflowY': 'auto'}, children=[
                        html.Div(id='welcome-message', style={'textAlign': 'center', 'paddingTop': '50px'},
                                 children=[html.P("Cliquez sur un point bleu ou une barre pour voir les détails.")]),
                        dcc.Graph(id='graphique-detail', style={'height': '350px'},
                                  config={'displayModeBar': False}),
                        html.Div(id='liste-competences',
                                 style={'marginTop': '20px', 'fontSize': '14px', 'borderTop': '1px solid #eee',
                                        'paddingTop': '10px'})
                    ])
                ])
            ]),

            html.Div(style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '15px',
                           'boxShadow': '0 4px 15px rgba(0,0,0,0.1)', 'marginBottom': '30px', 'marginTop': '30px'},
                     children=[
                         html.H3("Flux de carrière ", style={'textAlign': 'center', 'marginBottom': '20px'}),
                         html.Div(style={'width': '50%', 'margin': '0 auto 30px auto'}, children=[
                             dcc.Dropdown(
                                 id='dropdown-sankey',
                                 options=[{'label': 'Toutes les écoles', 'value': 'ALL'}] + [
                                     {'label': s, 'value': s} for s in sorted(df_polytech['Ecole'].unique())
                                 ],
                                 value='ALL', clearable=False
                             ),
                             html.Button("Réinitialiser la vue Polytech", id="btn-reset-vf", n_clicks=0,
                                         style={'marginTop': '10px', 'width': '100%'})
                         ]),
                         html.Div(id='sankey-headers',
                                  style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '0 40px',
                                         'marginBottom': '10px', 'borderBottom': '2px solid #f0f0f0'}),
                         dcc.Graph(id='sankey-graph')
                     ]),
            html.Div(style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': 'white',
                           'borderRadius': '15px'}, children=[
                html.H3("Analyse de Corrélation Globale", style={'textAlign': 'center'}),
                dcc.Graph(
                    id='competence-metier-scatter',
                    figure=px.scatter(
                        df_filtered, x='Nombre de Compétences', y='Nombre de Métiers',
                        color='Ecole', hover_name='Formation', size='Nombre de Compétences',
                        template="plotly_white"
                    )
                )
            ])
        ]),

        # ═══════════════════════════════════════════════════════════════
        # MAIN Layout Learnagement
        # ═══════════════════════════════════════════════════════════════
        html.Div(
            [
                html.Div(create_filter_section(), className="app-sidebar"),

                html.Div(
                    [
                        # KPIs
                        html.Div(
                            [
                                create_section_header("Vue d'ensemble de Polytech Annecy", "Indicateurs clés de votre curriculum"),
                                html.Div(id="stats-dashboard", className="stats-grid"),
                            ],
                            className="section",
                        ),

                        #  Analyses principales
                        html.Div(
                            [
                                create_section_header(
                                    "Analyses principales",
                                    "Statistiques par compétences ",
                                ),

                                # Bloc 1 : Compétences / AC + liste à droite
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            dcc.Graph(
                                                                id="apc-fig-comp-ac",
                                                                config={"displayModeBar": True, "displaylogo": False},
                                                            ),
                                                            style={"flex": "1"},
                                                        ),
                                                        html.Div(
                                                            id="apc-details-comp-ac",
                                                            style={
                                                                "width": "38%",
                                                                "maxHeight": "420px",
                                                                "overflowY": "auto",
                                                                "padding": "10px",
                                                                "borderLeft": "1px solid #eee",
                                                            },
                                                        )
                                                    ],
                                                    style={"display": "flex", "gap": "10px"}
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                    ],
                                    style={"marginBottom": "18px"},
                                ),

                                # Bloc 2 : Compétences / CE + liste à droite
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            dcc.Graph(
                                                                id="apc-fig-comp-ce",
                                                                config={"displayModeBar": True, "displaylogo": False},
                                                            ),
                                                            style={"flex": "1"},
                                                        ),
                                                        html.Div(
                                                            id="apc-details-comp-ce",
                                                            style={
                                                                "width": "38%",
                                                                "maxHeight": "420px",
                                                                "overflowY": "auto",
                                                                "padding": "10px",
                                                                "borderLeft": "1px solid #eee",
                                                            },
                                                        )
                                                    ],
                                                    style={"display": "flex", "gap": "10px"}
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                    ],
                                ),
                            ],
                            className="section",
                        ),

                        #  Orientation par métier
                        html.Div(
                            [
                                create_section_header(
                                    "Orientation par métier",
                                    "Choisissez un métier pour découvrir les compétences clés, les apprentissages critiques et leur progression par semestre.",
                                ),

                                html.Div(
                                    [
                                        html.Label("Métier", className="filter-label"),
                                        dcc.Dropdown(
                                            id="metier-sql",
                                            options=[{"label": m, "value": m} for m in metiers_sql],
                                            value=metiers_sql[0] if metiers_sql else None,
                                            clearable=False,
                                            className="filter-input",
                                        ),
                                    ],
                                    style={"maxWidth": "900px", "marginBottom": "12px"},
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="fig-metier-comp", config={"displayModeBar": True, "displaylogo": False})],
                                            className="card viz-card",
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="fig-metier-sem", config={"displayModeBar": True, "displaylogo": False})],
                                            className="card viz-card",
                                        ),
                                    ],
                                    className="viz-grid-2",
                                ),

                                html.Div(
                                    id="details-semestre-metier",
                                    className="card",
                                    style={"padding": "14px", "marginTop": "12px"},
                                ),
                            ],
                            className="section",
                        ),

                        # Mon parcours 
                        html.Div(
                            [
                                create_section_header("Mon parcours", "Exploration interactive par niveaux"),
                                html.Div(id="breadcrumb-nav", className="breadcrumb"),
                                html.Div(
                                    [
                                        html.Button(" Vue globale", id="btn-global", n_clicks=0, className="btn btn-primary"),
                                        html.Button("← Retour", id="btn-back", n_clicks=0, className="btn btn-outline"),
                                    ],
                                    className="drill-controls",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="viz1-timeline-drilldown",
                                            config={"displayModeBar": True, "displaylogo": False},
                                        )
                                    ],
                                    className="card viz-card",
                                ),
                            ],
                            className="section",
                        ),

                        # Autres analyses 
                        html.Div(
                            [
                                create_section_header("Autres analyses", ""),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz2-radar",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                    ],
                                    className="viz-grid-2",
                                ),
                            ],
                            className="section",
                        ),
                    ],
                    className="app-main-content",
                ),
            ],
            className="app-container",
        ),

        # Footer
        html.Div(
            [
                html.P("Utilisez les filtres pour adapter le dashboard à votre parcours", className="footer-text"),
                html.P("Learnagement-Approche par compétences, 2025", className="footer-credit"),
            ],
            className="app-footer",
        ),
    ],
    className="app-wrapper",
)

# ═══════════════════════════════════════════════════════════════════
# --- CALLBACKS POLYTECH 
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    Output('dropdown-sankey', 'value'),
    [Input({'type': 'marker-ecole', 'index': ALL}, 'n_clicks'), Input('btn-reset-vf', 'n_clicks')],
    prevent_initial_call=True
)
def sync_selection_vf(n_clicks_list, reset_clicks):
    ctx = callback_context
    if "btn-reset-vf" in ctx.triggered[0]['prop_id']:
        return "ALL"
    return json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']

@app.callback(Output("layer-markers", "children"), Input("layer-markers", "id"))
def render_markers_vf(_):
    return [
        dl.CircleMarker(
            center=coords,
            id={'type': 'marker-ecole', 'index': school},
            radius=10,
            color=COLORS['primary'],
            children=[dl.Tooltip(school)]
        ) for school, coords in coords_ecoles.items()
    ]

@app.callback(
    [Output('graphique-detail', 'figure'), Output('titre-ecole', 'children'), Output('welcome-message', 'style')],
    [Input('dropdown-sankey', 'value')]
)
def update_ui_bar_vf(selected_school):
    if selected_school == "ALL":
        return go.Figure(), "🏫 Sélectionnez une école", {'display': 'block'}
    df_school = df_filtered[df_filtered['Ecole'] == selected_school].sort_values('Nombre de Compétences')
    fig = px.bar(df_school, y='Formation', x='Nombre de Compétences', orientation='h',
                 color='Nombre de Compétences', color_continuous_scale='Viridis', template='plotly_white')
    fig.update_layout(margin=dict(l=150, r=20, t=20, b=40), xaxis=dict(dtick=1), coloraxis_showscale=False)
    return fig, selected_school, {'display': 'none'}

@app.callback([Output('sankey-graph', 'figure'), Output('sankey-headers', 'children')], Input('dropdown-sankey', 'value'))
def update_sankey_vf(selected_school):
    s_data = datavf if selected_school == 'ALL' else {selected_school: datavf.get(selected_school, [])}
    sources, targets, values, labels = [], [], [], []
    formations, ecoles, debouches = [], list(s_data.keys()), []

    for school, forms in s_data.items():
        for f in forms:
            f_name = reparer_texte(f.get('formation', '').replace('-', ' ').title())
            formations.append(f_name)
            combined = (f.get('secteurs', []) or []) + (f.get('metiers', []) or [])
            for item in combined:
                if isinstance(item, list):
                    item = item[0]
                if isinstance(item, str) and len(item) > 3:
                    debouches.append(reparer_texte(item.split(',')[0].split('(')[0].strip()[:35]))

    formations, debouches = sorted(list(set(formations))), sorted(list(set(debouches)))
    all_labels = formations + ecoles + debouches
    label_map = {name: i for i, name in enumerate(all_labels)}
    display_labels = [n if n not in debouches else "" for n in all_labels]

    for school, forms in s_data.items():
        for f in forms:
            f_idx = label_map[reparer_texte(f.get('formation', '').replace('-', ' ').title())]
            e_idx = label_map[school]
            sources.append(f_idx)
            targets.append(e_idx)
            values.append(1)
            combined = (f.get('secteurs', []) or []) + (f.get('metiers', []) or [])
            for item in combined:
                if isinstance(item, list):
                    item = item[0]
                if isinstance(item, str) and len(item) > 3:
                    clean = reparer_texte(item.split(',')[0].split('(')[0].strip()[:35])
                    sources.append(e_idx)
                    targets.append(label_map[clean])
                    values.append(1)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=50, thickness=20,
            label=display_labels,
            customdata=all_labels,
            hovertemplate='%{customdata}<extra></extra>',
            color="royalblue"
        ),
        link=dict(source=sources, target=targets, value=values, color="rgba(100, 150, 250, 0.2)")
    )])
    fig.update_layout(font_size=12, height=700, margin=dict(t=20, b=20, l=200, r=10))
    headers = [
        html.Div("FILIÈRE", style={'fontWeight': 'bold', 'width': '33%'}),
        html.Div("ÉCOLE", style={'fontWeight': 'bold', 'textAlign': 'center', 'width': '33%'}),
        html.Div("DÉBOUCHÉS", style={'fontWeight': 'bold', 'textAlign': 'right', 'width': '33%'})
    ]
    return fig, headers

@app.callback(
    Output('liste-competences', 'children'),
    [Input('graphique-detail', 'clickData'), Input('sankey-graph', 'clickData')],
    [State('sankey-graph', 'figure')]
)
def unified_click_handler_vf(barData, sankeyData, sankeyFig):
    ctx = callback_context
    if not ctx.triggered:
        return html.Em("Cliquez sur un élément pour voir le détail.")
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if input_id == 'graphique-detail' and barData:
        f_nom = barData['points'][0]['y']
        for s in datavf:
            for f in datavf[s]:
                if reparer_texte(f['formation'].replace('-', ' ').title()) == f_nom:
                    return html.Div([
                        html.B(f"Compétences pour {f_nom} :"),
                        html.Ul([html.Li(reparer_texte(c)) for c in f.get('competences', [])])
                    ])
    return "Aucun détail trouvé."


# ═══════════════════════════════════════════════════════════════════
# --- CALLBACKS LEARNAGEMENT (filtrage + parcours)
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    [Output("filtered-data-store", "data"), Output("filter-status", "children")],
    [
        Input("semester-filter", "value"),
        Input("ue-filter", "value"),
        Input("level-filter", "value"),
        Input("reset-filters", "n_clicks"),
    ],
    prevent_initial_call=False,
)
def filter_data(niveaux, competences, types_lien, reset_clicks):
    df = df_main
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "reset-filters.n_clicks":
        filtered_df = df.copy()
        status = f"{len(df)} objectifs affichés"
    else:
        filtered_df = df.copy()
        if niveaux:
            filtered_df = filtered_df[filtered_df["niveau"].isin(niveaux)]
        if competences:
            filtered_df = filtered_df[filtered_df["competence_label"].isin(competences)]
        if types_lien:
            filtered_df = filtered_df[filtered_df["type_lien"].isin(types_lien)]
        status = f"{len(filtered_df)} / {len(df)} objectifs"
    return filtered_df.to_json(date_format="iso", orient="split"), status

@app.callback(
    [
        Output("viz1-timeline-drilldown", "figure"),
        Output("drill-down-level", "data"),
        Output("selected-semester", "data"),
        Output("selected-module", "data"),
        Output("breadcrumb-nav", "children"),
    ],
    [
        Input("filtered-data-store", "data"),
        Input("viz1-timeline-drilldown", "clickData"),
        Input("btn-global", "n_clicks"),
        Input("btn-back", "n_clicks"),
    ],
    [
        State("drill-down-level", "data"),
        State("selected-semester", "data"),
        State("selected-module", "data"),
    ],
    prevent_initial_call=False,
)
def update_drilldown_viz1(
    filtered_data_json,
    clickData,
    btn_global_clicks,
    btn_back_clicks,
    current_level,
    current_semester,
    current_module,
):
    df = df_main
    ctx = dash.callback_context
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)

    filtered_df = filtered_df.assign(All_Competencies=filtered_df["niveau_code"])

    trigger_id = (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "none")

    new_level = current_level if current_level else "global"
    new_semester = current_semester
    new_module = current_module

    if trigger_id == "btn-global":
        new_level = "global"
        new_semester = None
        new_module = None
    elif trigger_id == "btn-back":
        if current_level == "module":
            new_level = "semester"
            new_module = None
        elif current_level == "semester":
            new_level = "global"
            new_semester = None
    elif trigger_id == "viz1-timeline-drilldown" and clickData and "points" in clickData:
        point = clickData["points"][0]
        if current_level in [None, "global"]:
            clicked_semester = point.get("y")
            if clicked_semester is not None:
                try:
                    if isinstance(clicked_semester, str):
                        clicked_semester = int(clicked_semester)
                except (ValueError, TypeError):
                    clicked_semester = None
                if clicked_semester is not None:
                    new_level = "semester"
                    new_semester = clicked_semester
                    new_module = None
        elif current_level == "semester":
            clicked_module_name = point.get("y")
            if clicked_module_name is not None:
                clicked_module = module_ids.get(clicked_module_name, clicked_module_name)
                if isinstance(clicked_module, str) and clicked_module.replace('.', '').isdigit():
                    clicked_module = float(clicked_module)
                new_level = "module"
                new_module = clicked_module

    try:
        df_pivot_global, df_pivot_semester, df_pivot_module = compute_competency_counts_per_module(filtered_df)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erreur: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig, "global", None, None, html.Div("Erreur")

    if new_level == "global":
        fig = create_heatmap_for_global(df_pivot_global)
        breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    elif new_level == "semester" and new_semester is not None:
        available_semesters = df_pivot_semester.index.get_level_values("niveau").unique().tolist()
        if new_semester in available_semesters:
            fig = create_heatmap_for_niveau(new_semester, df_pivot_semester, True)
            breadcrumb = html.Div([
                html.Span("Vue globale", className="breadcrumb-item"),
                html.Span(" / ", className="breadcrumb-separator"),
                html.Span(f"Semestre {new_semester}", className="breadcrumb-item active"),
            ])
        else:
            fig = create_heatmap_for_global(df_pivot_global)
            new_level = "global"
            new_semester = None
            breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    elif new_level == "module" and new_module is not None:
        available_modules = df_pivot_module.index.get_level_values("id_module").unique().tolist()
        if new_module in available_modules:
            fig = create_heatmap_for_module(new_module, df_pivot_module)
            module_name = module_names.get(new_module, f"Module {int(new_module)}")
            breadcrumb = html.Div([
                html.Span("Vue globale", className="breadcrumb-item"),
                html.Span(" / ", className="breadcrumb-separator"),
                html.Span(f"Semestre {new_semester}", className="breadcrumb-item"),
                html.Span(" / ", className="breadcrumb-separator"),
                html.Span(module_name, className="breadcrumb-item active"),
            ])
        else:
            fig = create_heatmap_for_global(df_pivot_global)
            new_level = "global"
            new_semester = None
            new_module = None
            breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    else:
        fig = create_heatmap_for_global(df_pivot_global)
        new_level = "global"
        new_semester = None
        new_module = None
        breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")

    return fig, new_level, new_semester, new_module, breadcrumb

@app.callback(Output("stats-dashboard", "children"), Input("filtered-data-store", "data"))
def update_stats(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df_main.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz7_statistics_dashboard(filtered_df)

@app.callback(Output("viz2-radar", "figure"), Input("filtered-data-store", "data"))
def update_viz2(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df_main.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz2_workload_radar(filtered_df)

# ═══════════════════════════════════════════════════════════════════
# Callbacks Analyses principales 
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    Output("apc-fig-comp-ac", "figure"),
    Output("apc-fig-comp-ce", "figure"),
    Input("filtered-data-store", "data"),
)
def update_apc_charts(filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df_main.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")

    if filtered_df.empty:
        return empty_fig("Aucune donnée après filtres"), empty_fig("Aucune donnée après filtres")

    # Graph 1 : nb AC par compétence
    df1 = (
        filtered_df.groupby("competence_label")["id_apprentissage_critique"]
        .nunique()
        .reset_index(name="nb_ac")
        .sort_values("nb_ac", ascending=True)
    )
    fig1 = bar_h_comp(df1, "nb_ac", "competence_label", "Compétences – Nombre d’Apprentissage critique (AC) associés")

    # Graph 2 : nb composantes essentielles par compétence 
    competences_actives = filtered_df["id_competence"].dropna().unique().tolist()
    ce_f = df_composantes[df_composantes["id_competence"].isin(competences_actives)].merge(
        df_competences[["id_competence", "code_competence"]],
        on="id_competence",
        how="left"
    ).rename(columns={"code_competence": "competence_label"})

    df2 = (
        ce_f.groupby("competence_label")
        .size()
        .reset_index(name="nb_ce")
        .sort_values("nb_ce", ascending=True)
    )
    fig2 = bar_h_comp(df2, "nb_ce", "competence_label", "Compétences – Nombre de composantes essentielles")

    return fig1, fig2


@app.callback(
    Output("apc-details-comp-ac", "children"),
    Input("apc-fig-comp-ac", "clickData"),
    State("filtered-data-store", "data"),
)
def show_ac_list(clickData, filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df_main.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")

    if not clickData:
        return "Clique sur une compétence pour afficher les apprentissages critiques."

    comp = clickData["points"][0]["y"]
    ac_list = (
        filtered_df[filtered_df["competence_label"] == comp][["libelle_apprentissage"]]
        .drop_duplicates()
        .sort_values("libelle_apprentissage")
    )
    if ac_list.empty:
        return f"Aucun apprentissage critique trouvé pour {comp}."

    return html.Div([
        html.H4(f"{comp} — Apprentissages critiques"),
        html.Ul([html.Li(r.libelle_apprentissage) for r in ac_list.itertuples(index=False)])
    ])


@app.callback(
    Output("apc-details-comp-ce", "children"),
    Input("apc-fig-comp-ce", "clickData"),
    State("filtered-data-store", "data"),
)
def show_ce_list(clickData, filtered_data_json):
    if filtered_data_json is None:
        filtered_df = df_main.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")

    if not clickData:
        return "Clique sur une compétence pour afficher les composantes essentielles."

    comp = clickData["points"][0]["y"]
    competences_actives = filtered_df["id_competence"].dropna().unique().tolist()

    ce_list = (
        df_composantes[df_composantes["id_competence"].isin(competences_actives)]
        .merge(df_competences[["id_competence", "code_competence"]], on="id_competence", how="left")
        .query("code_competence == @comp")[["libelle_composante_essentielle"]]
        .drop_duplicates()
        .sort_values("libelle_composante_essentielle")
    )

    if ce_list.empty:
        return f"Aucune composante essentielle trouvée pour {comp}."

    return html.Div([
        html.H4(f"{comp} — Composantes essentielles"),
        html.Ul([html.Li(r.libelle_composante_essentielle) for r in ce_list.itertuples(index=False)])
    ])

# ═══════════════════════════════════════════════════════════════════
# Callbacks Orientation par métier 
# ═══════════════════════════════════════════════════════════════════

@app.callback(
    Output("fig-metier-comp", "figure"),
    Output("fig-metier-sem", "figure"),
    Output("details-semestre-metier", "children"),
    Input("metier-sql", "value"),
    Input("fig-metier-sem", "clickData"),
    prevent_initial_call=False
)
def update_metier_sql(metier, clickData):
    if not metier:
        return empty_fig("Aucun métier sélectionné"), empty_fig("Aucun métier sélectionné"), "Choisis un métier."

    comp_ids = (
        df_situations[df_situations["libelle_situation"] == metier]["id_competence"]
        .dropna().unique().tolist()
    )

    dff_main = df_main[df_main["id_competence"].isin(comp_ids)].copy()
    if dff_main.empty:
        return empty_fig("Aucune donnée AC pour ce métier"), empty_fig("Aucune donnée"), "Aucune donnée."

    # Graph 1 : Compétences mobilisées 
    comp_job = (
        dff_main.groupby("competence_label")["id_apprentissage_critique"]
        .nunique()
        .reset_index(name="nb_ac")
        .sort_values("nb_ac", ascending=True)
    )
    fig_comp = bar_h(comp_job, "nb_ac", "competence_label", "Métier visé – Compétences mobilisées")

    # Graph 2 : Semestres 
    dff_sem = df_ac_sem[df_ac_sem["id_competence"].isin(comp_ids)].copy()
    dff_sem = dff_sem.dropna(subset=["id_semestre", "id_apprentissage_critique"])

    sem_job = (
        dff_sem.groupby("id_semestre")["id_apprentissage_critique"]
        .nunique()
        .reset_index(name="nb_ac")
        .sort_values("id_semestre")
    )
    fig_sem = px.bar(
        sem_job,
        x="id_semestre",
        y="nb_ac",
        text="nb_ac",
        title="Métier visé – Nombre d’AC par semestre"
    )
    fig_sem.update_traces(textposition="outside", cliponaxis=False)
    fig_sem.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white",
                          xaxis_title="Semestre", yaxis_title="Nombre d'AC")
    fig_sem.update_xaxes(dtick=1)

    if not clickData:
        return fig_comp, fig_sem, "Clique sur un semestre pour afficher les apprentissages critiques."

    try:
        sem = int(clickData["points"][0]["x"])
    except Exception:
        return fig_comp, fig_sem, "Clique sur une barre de semestre"

    ac_list = (
        dff_sem[dff_sem["id_semestre"] == sem][["code_competence", "libelle_apprentissage"]]
        .drop_duplicates()
        .sort_values(["code_competence", "libelle_apprentissage"])
    )
    if ac_list.empty:
        return fig_comp, fig_sem, f"Aucun apprentissage critique trouvé pour S{sem}."

    return fig_comp, fig_sem, html.Div([
        html.H4(f"Semestre S{sem} — Apprentissages critiques"),
        html.Ul([html.Li(f"{r.code_competence} — {r.libelle_apprentissage}") for r in ac_list.itertuples(index=False)])
    ])

# ═══════════════════════════════════════════════════════════════════
#                          LANCEMENT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
