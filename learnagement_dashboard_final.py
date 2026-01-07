"""
Learnagement Dashboard - Configuration et Traitement des Données
Dashboard basé sur la base de données learnagement.sql
Structure native APC (Approche Par Compétences)
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
from datetime import datetime
import networkx as nx
from io import StringIO
import sqlite3
import dash
from dash import dcc, html, Output, Input, State, ALL, callback_context
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
    # Ajout de la police Poppins pour la partie Polytech
    external_stylesheets=['https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap']
)

# ═══════════════════════════════════════════════════════════════════
#                        PARTIE RESEAU POLYTECH (Section 1)
# ═══════════════════════════════════════════════════════════════════

# --- 1. FONCTIONS DE NETTOYAGE ---

def reparer_texte(texte):
    if not isinstance(texte, str): return texte
    try:
        return texte.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            return texte.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texte

def count_competences(competences):
    if not isinstance(competences, list): return 0
    if len(competences) > 0 and isinstance(competences[0], str) and "Erreur" in competences[0]:
        return 0
    return len(competences)

def count_metiers(metiers):
    if not isinstance(metiers, list): return 0
    flat_metiers = []
    for item in metiers:
        if isinstance(item, list): flat_metiers.extend(item)
        elif isinstance(item, str): flat_metiers.append(item)
    return len([m for m in flat_metiers if isinstance(m, str) and len(m) > 3])

# --- 2. CHARGEMENT ET PRÉPARATION DES DONNÉES ---

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

df = pd.DataFrame(records)
df_filtered = df[df['Nombre de Compétences'] > 0].copy()
# ═══════════════════════════════════════════════════════════════════
#                    CHARGEMENT DES DONNÉES SQL
# ═══════════════════════════════════════════════════════════════════

def load_learnagement_data(sql_file_path):
    """
    Charge les données depuis learnagement.sql et crée les DataFrames principaux
    Structure native de la base de données APC
    """
    import re
    
    # Lire le fichier SQL
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sql_content = f.read()
    
    def parse_insert_values(table_name, sql_text):
        """Parse les INSERT statements d'une table"""
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
    
    # ═══════════════════════════════════════════════════════════════
    # Extraire les données des tables principales
    # ═══════════════════════════════════════════════════════════════
    
    # Table 1: Compétences
    competences_data = parse_insert_values('APC_competence', sql_content)
    df_competences = pd.DataFrame(
        competences_data,
        columns=['id_competence', 'libelle_competence', 'code_competence', 'description']
    )
    
    # Table 2: Niveaux (3 niveaux par compétence = 12 niveaux au total)
    niveaux_data = parse_insert_values('APC_niveau', sql_content)
    df_niveaux = pd.DataFrame(
        niveaux_data,
        columns=['id_niveau', 'id_competence', 'niveau', 'libelle_niveau']
    )
    
    # Table 3: Apprentissages Critiques (AC)
    ac_data = parse_insert_values('APC_apprentissage_critique', sql_content)
    df_apprentissages = pd.DataFrame(
        ac_data,
        columns=['id_apprentissage_critique', 'id_niveau', 'libelle_apprentissage']
    )
    
    # Table 4: Liaison AC <-> Modules
    ac_modules_data = parse_insert_values('APC_apprentissage_critique_as_module', sql_content)
    df_ac_modules = pd.DataFrame(
        ac_modules_data,
        columns=['id_apprentissage_critique', 'id_module', 'type_lien']
    )
    
    # Table 5: Modules (pour les noms et codes)
    modules_data = parse_insert_values('MAQUETTE_module', sql_content)
    df_modules = pd.DataFrame(
        modules_data,
        columns=['id_module', 'code_module', 'nom', 'ECTS', 'id_discipline', 'id_semestre', 
                 'hCM', 'hTD', 'hTP', 'hTPTD', 'hPROJ', 'hPersonnelle', 'id_responsable', 'commentaire']
    )
    
    # Table 6: Composantes essentielles
    composantes_data = parse_insert_values('APC_composante_essentielle', sql_content)
    df_composantes = pd.DataFrame(
        composantes_data,
        columns=['id_composante_essentielle', 'id_competence', 'libelle_composante_essentielle']
    )
    
    return df_competences, df_niveaux, df_apprentissages, df_ac_modules, df_modules, df_composantes


# Charger les données
df_competences, df_niveaux, df_apprentissages, df_ac_modules, df_modules, df_composantes = load_learnagement_data(
    "learnagement.sql"
)

# ═══════════════════════════════════════════════════════════════════
#                    TRAITEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#                    TRAITEMENT DES DONNÉES - VERSION UNIFIÉE
# ═══════════════════════════════════════════════════════════════════

# Créer le DataFrame de BASE avec TOUS les AC (une ligne par AC)
df_base = (
    df_apprentissages
    .merge(df_niveaux, on='id_niveau', how='left')
    .merge(df_competences, on='id_competence', how='left')
)

# Pour chaque AC, récupérer ses modules (mais garder une seule ligne par AC)
# On va créer une colonne avec la liste des modules
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

# Joindre avec la base
df_main = df_base.merge(ac_modules, on='id_apprentissage_critique', how='left')

# Pour les AC sans module, créer des listes vides
df_main['modules_list'] = df_main['modules_list'].apply(lambda x: x if isinstance(x, list) else [])
df_main['types_lien_list'] = df_main['types_lien_list'].apply(lambda x: x if isinstance(x, list) else [])

# Ajouter des colonnes pratiques
df_main['nb_modules'] = df_main['modules_list'].apply(len)
df_main['has_module'] = df_main['nb_modules'] > 0

# Pour compatibilité avec le code existant, prendre le premier module si existe
df_main['id_module'] = df_main['modules_list'].apply(lambda x: x[0] if len(x) > 0 else 0)
df_main['type_lien'] = df_main['types_lien_list'].apply(lambda x: x[0] if len(x) > 0 else 'Non associé')

# Ajouter des colonnes calculées utiles pour les visualisations

# 1. Utiliser directement les codes de compétence (COMP_IDU1, COMP_IDU2, etc.)
df_main['competence_label'] = df_main['code_competence']  # Garder les codes tels quels

# Dictionnaire pour les mappings (même si on utilise les codes directement)
competence_labels = {
    'COMP_IDU1': 'COMP_IDU1',
    'COMP_IDU2': 'COMP_IDU2',
    'COMP_IDU3': 'COMP_IDU3',
    'COMP_IDU4': 'COMP_IDU4'
}

# 2. Code niveau complet (ex: "Réaliser-N1", "Optimiser-N2")
df_main['niveau_code'] = df_main['competence_label'] + '-N' + df_main['niveau'].astype(str)

# Créer un dictionnaire pour les noms de modules
module_names = {}
for _, row in df_modules.iterrows():
    module_id = row['id_module']
    code = row['code_module'] if pd.notna(row['code_module']) else f"M{module_id}"
    nom = row['nom'] if pd.notna(row['nom']) else "Module"
    module_names[module_id] = f"{code} - {nom}"

# Ajouter le module virtuel
module_names[0] = "Non associé"
module_names[0.0] = "Non associé"

# Créer un dictionnaire inverse pour retrouver l'ID à partir du nom
module_ids = {name: id_mod for id_mod, name in module_names.items()}


# 3. Semestre approximatif basé sur le niveau (N1=S1-S2, N2=S3-S4, N3=S5-S6)
semestre_map = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6]
}
df_main['semestres'] = df_main['niveau'].map(lambda x: semestre_map.get(x, []))

# 4. Catégorie de type de lien
df_main['lien_requis'] = df_main['type_lien'] == 'Requis'


# ═══════════════════════════════════════════════════════════════════

# Identifier les AC sans module associé
mask_sans_module = df_main['id_module'].isna()
nb_ac_sans_module = mask_sans_module.sum()

if nb_ac_sans_module > 0:
    print(f"\n  {nb_ac_sans_module} apprentissages critiques sans module associé détectés")
    
    # Répartition par compétence
    ac_sans_module_comp = df_main[mask_sans_module].groupby('code_competence').size()
    print("   Répartition par compétence :")
    for comp, count in ac_sans_module_comp.items():
        comp_label = competence_labels.get(comp, comp)
        print(f"   - {comp_label}: {count} AC")
    
    # Assigner un module virtuel pour les inclure dans toutes les visualisations
    df_main.loc[mask_sans_module, 'id_module'] = 0.0  # Module 0 = "Sans module"
    df_main.loc[mask_sans_module, 'type_lien'] = 'Non associé'
    
    print(f" Ces AC sont maintenant inclus avec le module virtuel '0 - Non associé'")

# ═══════════════════════════════════════════════════════════════════
#                      DESIGN SYSTEM COULEURS
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    # Couleurs principales
    "primary": "#3B82F6",
    "secondary": "#8B5CF6",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#06B6D4",
    "light": "#F8FAFC",
    "dark": "#1E293B",
    
    # Couleurs par compétence (4 compétences BUT Informatique)
    "competences": {
        "COMP_IDU1": "#FF513F",  # Rouge - Concevoir et mettre en œuvre des systèmes informatiques
        "COMP_IDU2": "#FFA03F",  # Orange - Collecter et traiter des données numériques
        "COMP_IDU3": "#3B82F6",  # Bleu - Gérer les usages des données numériques
        "COMP_IDU4": "#10B981",  # Vert - Gérer un projet informatique
    },
    
    # Couleurs par niveau
    "niveaux": {
        1: "#BFDBFE",  # Bleu clair - Niveau 1
        2: "#60A5FA",  # Bleu moyen - Niveau 2
        3: "#2563EB",  # Bleu foncé - Niveau 3
    },
    
    # Couleurs par type de lien module
    "type_lien": {
        "Requis": "#EF4444",           # Rouge
        "Recommandé": "#F59E0B",       # Orange
        "Complémentaire": "#10B981",   # Vert
    },
}

def get_competence_color(competence_label):
    """Retourne la couleur associée à une compétence"""
    return COLORS["competences"].get(competence_label, "#777777")

def get_niveau_color(niveau):
    """Retourne la couleur associée à un niveau"""
    return COLORS["niveaux"].get(niveau, "#AAAAAA")

# ═══════════════════════════════════════════════════════════════════
#                    FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════

def get_ac_by_competence(competence_label):
    """Retourne tous les AC d'une compétence donnée"""
    return df_main[df_main['competence_label'] == competence_label]

def get_ac_by_niveau(competence_label, niveau):
    """Retourne tous les AC d'une compétence à un niveau donné"""
    return df_main[
        (df_main['competence_label'] == competence_label) & 
        (df_main['niveau'] == niveau)
    ]

def count_ac_per_module(df=None):
    """Compte le nombre d'AC par module"""
    if df is None:
        df = df_main
    return (
        df.groupby('id_module')
        .agg({
            'id_apprentissage_critique': 'count',
            'libelle_apprentissage': lambda x: list(x)
        })
        .rename(columns={'id_apprentissage_critique': 'nb_ac'})
    )

def count_modules_per_competence_niveau():
    """Compte le nombre de modules par compétence et niveau"""
    return (
        df_main.groupby(['competence_label', 'niveau'])
        .agg({
            'id_module': 'nunique',
            'id_apprentissage_critique': 'count'
        })
        .rename(columns={
            'id_module': 'nb_modules',
            'id_apprentissage_critique': 'nb_ac'
        })
    )

# ═══════════════════════════════════════════════════════════════════
#                    MATRICES PIVOT POUR VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════

def create_competence_niveau_matrix():
    """
    Crée une matrice Compétence × Niveau avec le nombre d'AC
    Utilisé pour les heatmaps et visualisations globales
    """
    pivot = df_main.pivot_table(
        index='competence_label',
        columns='niveau',
        values='id_apprentissage_critique',
        aggfunc='count',
        fill_value=0
    )
    return pivot

def create_module_competence_matrix():
    """
    Crée une matrice Module × Compétence avec le nombre d'AC
    Utilisé pour voir la couverture des compétences par module
    """
    pivot = df_main.pivot_table(
        index='id_module',
        columns='competence_label',
        values='id_apprentissage_critique',
        aggfunc='count',
        fill_value=0
    )
    return pivot

def parse_competencies(row):
    """Pour compatibilité - extrait les compétences d'une ligne"""
    comps = []
    if 'niveau_code' in row.index and pd.notna(row['niveau_code']):
        comps.append(row['niveau_code'])
    return comps


# ═══════════════════════════════════════════════════════════════════
#           PIVOTS POUR HEATMAPS (3 niveaux de drill-down)
# ═══════════════════════════════════════════════════════════════════

def compute_competency_counts_per_module(df_input):
    """
    Calcule les matrices pivot pour les 3 niveaux de visualisation interactive:
    
    Niveau 1 - GLOBAL : Niveau × Compétence (vue d'ensemble)
    Niveau 2 - NIVEAU : (Niveau, Module) × Compétence (zoom sur un niveau)
    Niveau 3 - MODULE : (Niveau, Module, AC) × Compétence (détail d'un module)
    
    Équivalent de la fonction Excel mais adaptée à notre structure SQL
    
    Args:
        df_input: DataFrame à utiliser (par défaut df_main)
    
    Returns:
        tuple: (df_pivot_global, df_pivot_niveau, df_pivot_module)
    """
    df_work = df_input.copy()
    
    # S'assurer qu'on a une colonne pour les apprentissages critiques
    if "libelle_apprentissage" not in df_work.columns:
        df_work = df_work.assign(libelle_apprentissage=df_work.index.astype(str))
    
    # Créer une colonne avec la liste des compétences (ici une seule par ligne)
    # On utilise niveau_code comme identifiant de compétence (ex: "Réaliser-N1")
    df_work = df_work.assign(All_Competencies=df_work["niveau_code"])
    
    # Pas besoin d'explode car on a déjà une compétence par ligne
    df_exploded = df_work.copy()
    
    # ════════════════════════════════════════════════════════════════
    # NIVEAU 1 - PIVOT GLOBAL : Niveau × Compétence
    # Vue d'ensemble : combien d'AC par niveau et par compétence
    # ════════════════════════════════════════════════════════════════
    df_pivot_global = (
        df_exploded
        .groupby(["niveau", "competence_label"], as_index=False)
        .size()
        .pivot(index="niveau", columns="competence_label", values="size")
        .fillna(0)
        .astype("Float64")
    )
    
    # ════════════════════════════════════════════════════════════════
    # NIVEAU 2 - PIVOT NIVEAU : (Niveau, Module) × Compétence
    # Zoom sur un niveau : combien d'AC par module et par compétence
    # ════════════════════════════════════════════════════════════════
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
    
    # ════════════════════════════════════════════════════════════════
    # NIVEAU 3 - PIVOT MODULE : (Niveau, Module, AC) × Compétence
    # Détail d'un module : type de lien pour chaque AC et compétence
    # ════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════
#                       HEATMAPS DRILL-DOWN
# ═══════════════════════════════════════════════════════════════════

def create_heatmap_for_global(df_pivot_global):
    """
    Crée la heatmap de niveau 1 : Vue globale Niveau × Compétence
    Permet de cliquer sur un niveau pour voir ses modules
    """
    fig = go.Figure()
    
    # Créer une trace par compétence avec sa couleur spécifique
    for j in range(df_pivot_global.shape[1]):
        comp = df_pivot_global.columns[j]
        color_comp = get_competence_color(comp)
        color_scale = ["white", color_comp]
        
        # Créer une copie où on masque les autres compétences
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
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Inter, sans-serif"),
        ),
        xaxis_title="Compétences",
        xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
        yaxis_title="Niveaux",
        yaxis=dict(
            type="category",
            autorange="reversed",
            showgrid=False,
            tickfont=dict(size=12),
        ),
        height=600,
        font=dict(size=11, family="Inter, sans-serif"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=100, r=60, t=120, b=150),
        clickmode="event+select",
    )

    return fig


def create_heatmap_for_niveau(niveau, df_pivot_niveau, ylegend=True):
    """
    Crée la heatmap de niveau 2 : Vue Niveau → (Module × Compétence)
    Montre les modules d'un niveau spécifique
    Permet de cliquer sur un module pour voir ses AC
    """
    # Extraire les données du niveau sélectionné
    df_used = df_pivot_niveau.xs(niveau, level="niveau")

    fig = go.Figure()
    z_min = 0
    z_max = df_pivot_niveau.max().max()

    # Créer une trace par compétence
    for j in range(df_used.shape[1]):
        comp = df_used.columns[j]
        color_comp = get_competence_color(comp)
        color_scale = ["white", color_comp]
        
        # Masquer les autres compétences
        df_comp = df_used.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA
        
        # Créer les noms de modules pour le hover
        module_names_list = [module_names.get(mod, f"Module {mod}") for mod in df_used.index]
        
        fig.add_trace(
            go.Heatmap(
                z=df_comp.values,
                x=df_used.columns,
                y=module_names_list,  # Utiliser les noms au lieu des IDs
                zmin=z_min,
                zmax=z_max,
                colorscale=color_scale,
                showscale=False,
                hoverongaps=False,
                xgap=1,
                ygap=1,
                hovertemplate="<b>%{y}</b><br>%{x}<br>AC: %{z}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Modules du niveau {niveau}<br><sub>Cliquez sur un module pour voir les apprentissages critiques</sub>",
            x=0.5,
            xanchor="center",
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
    """
    Crée la heatmap de niveau 3 : Vue Module → (AC × Compétence)
    Montre les apprentissages critiques d'un module
    Affiche le type de lien (Requis, Recommandé, Complémentaire)
    """
    # Convertir les types de lien en valeurs numériques pour la visualisation
    df_reorganized = df_pivot_module.replace({
        "Requis": 3,
        "Recommandé": 2, 
        "Complémentaire": 1,
        "": 0
    })
    
    # Extraire les données du module sélectionné
    df_used = (
        df_reorganized
        .xs(module_id, level="id_module")
        .astype("Int64")
    )

    fig = go.Figure()
    z_min = 0
    z_max = 3  # 3 = Requis (max importance)

    # Créer une trace par compétence
    for j in range(df_used.shape[1]):
        comp = df_used.columns[j]
        color_comp = get_competence_color(comp)
        color_scale = ["white", color_comp]
        
        # Masquer les autres compétences
        df_comp = df_used.copy()
        for col in df_comp.columns:
            if col != comp:
                df_comp.loc[:, col] = pd.NA
        
        # Créer un texte personnalisé pour le hover
        hover_text = []
        for idx in df_used.index:
            if hasattr(idx, '__len__') and len(idx) > 1:
                # Multi-index avec libelle_apprentissage
                ac_text = idx[-1] if isinstance(idx, tuple) else idx
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
                xgap=1,
                ygap=1,
                customdata=df_comp.values,
                hovertemplate="<b>%{y}</b><br>%{x}<br>Type: %{customdata}<extra></extra>",
            )
        )

    # Trouver le niveau du module pour le titre
    if hasattr(df_used.index, 'get_level_values'):
        niveau_module = df_used.index.get_level_values('niveau')[0] if 'niveau' in df_used.index.names else "?"
    else:
        niveau_module = "?"

    module_name = module_names.get(module_id, f"Module {module_id}")
    fig.update_layout(
        title=dict(
            text=f"{module_name} (Niveau {niveau_module}) - Apprentissages critiques",
            x=0.5,
            xanchor="center",
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
    """Crée une carte statistique"""
    content = [
        html.Div([
            html.Div(title, className="stat-label"),
        ], className="stat-header"),
        html.Div(str(value), className="stat-value", style={"color": color}),
    ]
    if trend is not None:
        trend_class = "stat-trend-up" if trend > 0 else "stat-trend-down"
        content.append(
            html.Div(
                [
                    html.Span("↑" if trend > 0 else "↓"),
                    html.Span(f"{abs(trend)}%"),
                ],
                className=f"stat-trend {trend_class}",
            )
        )
    if comparison:
        content.append(html.Div(comparison, className="stat-description"))
    else:
        content.append(html.Div(subtitle, className="stat-description"))

    return html.Div(content, className="card stat-card card-content")


def create_section_header(title, subtitle):
    """Crée un en-tête de section"""
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
    """Crée la section de filtres adaptée pour Learnagement"""
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Filtres", className="sidebar-title"),
                    html.P(
                        "Personnalisez votre vue",
                        className="sidebar-subtitle",
                    ),
                ],
                className="sidebar-header",
            ),
            html.Div(
                [
                    # Niveaux
                    html.Div(
                        [
                            html.Label(
                                "Niveaux",
                                className="filter-label",
                            ),
                            dcc.Dropdown(
                                id="semester-filter",
                                options=[
                                    {"label": f"Niveau {n}", "value": n}
                                    for n in sorted(df_main["niveau"].dropna().unique())
                                ],
                                value=sorted(df_main["niveau"].dropna().unique()),
                                multi=True,
                                className="filter-input",
                                placeholder="Tous les niveaux",
                            ),
                        ],
                        className="filter-group",
                    ),
                    # Compétences
                    html.Div(
                        [
                            html.Label(
                                "Compétences",
                                className="filter-label",
                            ),
                            dcc.Dropdown(
                                id="ue-filter",
                                options=[
                                    {"label": comp, "value": comp}
                                    for comp in sorted(df_main["competence_label"].dropna().unique())
                                ],
                                value=sorted(df_main["competence_label"].dropna().unique()),
                                multi=True,
                                className="filter-input",
                                placeholder="Toutes les compétences",
                            ),
                        ],
                        className="filter-group",
                    ),
                    # Types de lien
                    html.Div(
                        [
                            html.Label(
                                "Types de lien",
                                className="filter-label",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="level-filter",
                                                options=[
                                                    {"label": html.Div([
                                                        html.Span("R", className="level-badge level-e"),
                                                        html.Span("Requis", className="level-text")
                                                    ], className="level-option"), "value": "Requis"},
                                                    {"label": html.Div([
                                                        html.Span("Rec", className="level-badge level-m"),
                                                        html.Span("Recommandé", className="level-text")
                                                    ], className="level-option"), "value": "Recommandé"},
                                                    {"label": html.Div([
                                                        html.Span("C", className="level-badge level-n"),
                                                        html.Span("Complémentaire", className="level-text")
                                                    ], className="level-option"), "value": "Complémentaire"},
                                                    {"label": html.Div([
                                                        html.Span("N/A", className="level-badge level-na"),
                                                        html.Span("Non associé", className="level-text")
                                                    ], className="level-option"), "value": "Non associé"},
                                                ],
                                                value=["Requis", "Recommandé", "Complémentaire", "Non associé"],
                                                className="level-checklist",
                                            ),
                                        ],
                                    ),
                                ],
                                className="level-filter-container",
                            ),
                        ],
                        className="filter-group",
                    ),
                    html.Div(
                        [
                            html.Button(
                                [
                                    html.Span("↻", style={"marginRight": "6px", "fontSize": "14px"}),
                                    "Réinitialiser"
                                ],
                                id="reset-filters",
                                className="btn btn-reset",
                            ),
                        ],
                        className="filter-footer",
                    ),
                    html.Div(
                        id="filter-status",
                        className="filter-status",
                    ),
                ],
                className="sidebar-content",
            ),
        ],
        className="sidebar-panel",
    )


def viz2_workload_radar(filtered_df):
    """Radar chart : Charge de travail par niveau"""
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
        requis_count = (niv_data["type_lien"] == "Requis").sum()
        
        workload_data.append({
            "Niveau": f"N{niv}",
            "Compétences": unique_comps,
            "AC": total_ac,
            "Modules": modules_count,
            "Requis": requis_count,
        })
    
    wl_df = pd.DataFrame(workload_data)
    fig = go.Figure()
    
    for metric in ["Compétences", "AC", "Modules", "Requis"]:
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
        title="Charge de travail par niveau",
        polar=dict(radialaxis=dict(visible=True)),
        height=650,
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def viz3_competency_heatmap(filtered_df):
    """Bar Chart : Composantes Essentielles par Compétence"""
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune donnée à afficher.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(paper_bgcolor="white", height=500)
        return fig
    
    # Filtrer les composantes essentielles en fonction des compétences actives
    competences_actives = filtered_df['id_competence'].unique()
    composantes_filtrees = df_composantes[df_composantes['id_competence'].isin(competences_actives)]
    
    # Joindre avec df_competences pour avoir les labels
    composantes_avec_labels = composantes_filtrees.merge(
        df_competences[['id_competence', 'code_competence']], 
        on='id_competence',
        how='left'
    )
    
    # Mapper les codes vers les labels
    composantes_avec_labels['competence_label'] = composantes_avec_labels['code_competence'].map(competence_labels)
    
    # Compter par compétence
    comp_counts = composantes_avec_labels.groupby('competence_label').size().reset_index(name='count')
    comp_counts = comp_counts.sort_values('count', ascending=True)
    
    # Créer le bar chart VERTICAL
    fig = go.Figure()
    
    colors = [get_competence_color(comp) for comp in comp_counts['competence_label']]
    
    fig.add_trace(go.Bar(
        x=comp_counts['competence_label'],
        y=comp_counts['count'],
        marker=dict(color=colors),
        text=comp_counts['count'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Composantes: %{y}<extra></extra>',
    ))
    
    fig.update_layout(
        title="Composantes Essentielles par Compétence",
        xaxis_title="Compétences",
        yaxis_title="Nombre de Composantes Essentielles",
        height=450,
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(l=60, r=50, t=80, b=100),
        xaxis=dict(tickangle=-45),
    )
    
    return fig
    """Diagramme Sankey : Flux Compétences → Modules"""
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée à afficher.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="white", height=600)
        return fig
    
    # Créer les nœuds et liens
    comp_counts = filtered_df.groupby('competence_label').size().to_dict()
    module_counts = filtered_df.groupby('id_module').size().to_dict()
    
    # Labels
    comp_labels = list(comp_counts.keys())
    module_labels = [f"Module {int(m)}" for m in module_counts.keys()]
    all_labels = comp_labels + module_labels
    
    # Couleurs
    comp_colors = [get_competence_color(c) for c in comp_labels]
    module_colors = ["#E5E7EB"] * len(module_labels)
    all_colors = comp_colors + module_colors
    
    # Links
    source = []
    target = []
    value = []
    
    for _, row in filtered_df.iterrows():
        comp = row['competence_label']
        mod = row['id_module']
        if pd.notna(comp) and pd.notna(mod):
            source.append(all_labels.index(comp))
            target.append(all_labels.index(f"Module {int(mod)}"))
            value.append(1)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=0.5),
            label=all_labels,
            color=all_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(0,0,0,0.2)"
        )
    )])
    
    fig.update_layout(
        title="Flux : Compétences → Modules",
        font=dict(size=12, family="Inter, sans-serif"),
        height=600,
        paper_bgcolor="white",
    )
    return fig


def viz5_critical_competencies(filtered_df):
    """Bar chart : Compétences critiques"""
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée à afficher.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="white", height=500)
        return fig
    
    # Compter les AC par compétence
    comp_counts = filtered_df.groupby('competence_label').size().reset_index(name='count')
    comp_counts = comp_counts.sort_values('count', ascending=True)
    
    # Couleurs
    colors = [get_competence_color(c) for c in comp_counts['competence_label']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=comp_counts['count'],
            y=comp_counts['competence_label'],
            orientation='h',
            marker=dict(color=colors),
            text=comp_counts['count'],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>AC: %{x}<extra></extra>',
        )
    ])
    
    fig.update_layout(
        title="Nombre d'apprentissages critiques par compétence",
        xaxis_title="Nombre d'AC",
        yaxis_title="",
        height=500,
        font=dict(size=12, family="Inter, sans-serif"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig




def create_competence_legend_card():
    """Carte affichant la légende des codes de compétences"""
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
                html.Div(
                    style={
                        "width": "20px",
                        "height": "20px",
                        "backgroundColor": comp["color"],
                        "borderRadius": "4px",
                        "marginRight": "12px",
                        "flexShrink": "0"
                    }
                ),
                html.Div([
                    html.Span(comp["code"], style={"fontWeight": "600", "marginRight": "8px"}),
                    html.Span(comp["label"], style={"color": "#6B7280", "fontSize": "14px"}),
                ]),
            ], style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "12px",
                "padding": "8px",
                "backgroundColor": "#F9FAFB",
                "borderRadius": "6px",
            })
        )
    
    return html.Div([
        html.Div([
            html.H3(" Légende des Compétences", style={
                "margin": "0",
                "fontSize": "18px",
                "fontWeight": "600",
                "color": "#1F2937"
            }),
        ], style={"marginBottom": "20px"}),
        html.Div(legend_items),
    ], style={
        "backgroundColor": "white",
        "padding": "24px",
        "borderRadius": "12px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
        "marginBottom": "30px",
    })


def viz7_statistics_dashboard(filtered_df):
    """Dashboard de statistiques (KPI cards)"""
    if filtered_df.empty:
        return html.Div("Aucune donnée", style={"textAlign": "center", "padding": "20px"})
    
    # Calculer les statistiques
    # Compter les AC UNIQUES, pas les lignes (duplications AC-Module)
    total_ac = filtered_df["id_apprentissage_critique"].nunique()
    total_modules = sum(filtered_df['nb_modules']) if 'nb_modules' in filtered_df.columns else filtered_df['id_module'].nunique()
    total_competences = filtered_df['competence_label'].nunique()
    requis_pct = (filtered_df['type_lien'] == 'Requis').sum() / total_ac * 100 if total_ac > 0 else 0
    
    # Filtrer les composantes essentielles en fonction des compétences actives
    competences_actives = filtered_df['id_competence'].unique()
    composantes_filtrees = df_composantes[df_composantes['id_competence'].isin(competences_actives)]
    total_composantes = len(composantes_filtrees)
    nb_comp_avec_composantes = composantes_filtrees['id_competence'].nunique()
    
    return html.Div(
        [
            html.Div(
                [
                    create_stat_card(
                        "Apprentissages Critiques",
                        total_ac,
                        f"{total_modules} modules ({len(filtered_df)} associations)",
                        COLORS["primary"]
                    ),
                    create_stat_card(
                        "Compétences",
                        total_competences,
                        "Compétences distinctes",
                        COLORS["success"]
                    ),
                    create_stat_card(
                        "Modules",
                        total_modules,
                        "Modules uniques",
                        COLORS["warning"]
                    ),
                    create_stat_card(
                        "Composantes Essentielles",
                        total_composantes,
                        f"Réparties sur {nb_comp_avec_composantes} compétence(s)",
                        COLORS["danger"]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                    "gap": "20px",
                    "marginBottom": "30px"
                }
            ),
            # Carte de légende des compétences
            create_competence_legend_card(),
        ]
    )

    """Graphe de réseau 3D : Modules ↔ Compétences"""
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée à afficher.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="white", height=700)
        return fig
    
    # Créer le graphe
    G = nx.Graph()
    
    # Ajouter les nœuds
    for comp in filtered_df['competence_label'].unique():
        if pd.notna(comp):
            G.add_node(comp, node_type='competence')
    
    for mod in filtered_df['id_module'].unique():
        if pd.notna(mod):
            G.add_node(f"M{int(mod)}", node_type='module')
    
    # Ajouter les arêtes
    for _, row in filtered_df.iterrows():
        if pd.notna(row['competence_label']) and pd.notna(row['id_module']):
            G.add_edge(row['competence_label'], f"M{int(row['id_module'])}")
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42, dim=3)
    
    # Extraire les positions
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='#CBD5E1', width=2),
        hoverinfo='none'
    )
    
    # Nœuds
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    node_size = []
    
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(node)
        
        if G.nodes[node]['node_type'] == 'competence':
            node_color.append(get_competence_color(node))
            node_size.append(15)
        else:
            node_color.append('#94A3B8')
            node_size.append(8)
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate='<b>%{text}</b><extra></extra>',
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Réseau 3D : Compétences ↔ Modules",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        height=700,
        paper_bgcolor="white",
    )
    return fig

#############Fonctions POLYTECH RESEAU#######################

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
                                        html.H1(
                                            "Learnagement - Tableau de bord",
                                            className="header-title",
                                        ),
                                        html.P(
                                            "Visualisez où, quand et comment développer vos compétences",
                                            className="header-subtitle",
                                        ),
                                    ]
                                ),
                            ],
                            className="header-logo",
                        ),
                        html.Div(
                            f"Mise à jour : {datetime.now().strftime('%d/%m/%Y')}",
                            className="header-actions",
                        ),
                    ],
                    className="header-content",
        )
    ],
    className="app-header",
    style={
        "position": "sticky",  # Garde le header en haut lors du scroll
        "top": "0",            # Colle le header au sommet de la page
        "zIndex": "9999",      # Place le header au-dessus de la carte Leaflet
        "backgroundColor": "white", # Assure que le header n'est pas transparent
        "width": "100%"        # Prend toute la largeur
    }
),
        #######################################partie polytech reseeau##############

        html.Div(style={'padding': '20px', 'fontFamily': '"Poppins", sans-serif'}, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'}, children=[
                html.Div(style={'flex': '1.5'}, children=[
                    html.H3("Localisation des écoles du Reseau Polytech "),
                    dl.Map([dl.TileLayer(), dl.LayerGroup(id="layer-markers")], center=[46.5, 2.5], zoom=6, style={'height': '65vh', 'borderRadius': '12px'})
                ]),
                html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'height': '65vh'}, children=[
                    html.H3(id='titre-ecole', children="🏫 Sélectionnez une école", style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.Div(style={'flex': '1', 'backgroundColor': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.05)', 'padding': '15px', 'overflowY': 'auto'}, children=[
                        html.Div(id='welcome-message', style={'textAlign': 'center', 'paddingTop': '50px'}, children=[html.P("Cliquez sur un point bleu ou une barre pour voir les détails.")]),
                        dcc.Graph(id='graphique-detail', style={'height': '350px'}, config={'displayModeBar': False}),
                        html.Div(id='liste-competences', style={'marginTop': '20px', 'fontSize': '14px', 'borderTop': '1px solid #eee', 'paddingTop': '10px'})
                    ])
                ])
            ]),

            html.Div(style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)', 'marginBottom': '30px', 'marginTop': '30px'}, children=[
                html.H3("Flux de carrière ", style={'textAlign': 'center', 'marginBottom': '20px'}),
                html.Div(style={'width': '50%', 'margin': '0 auto 30px auto'}, children=[
                    dcc.Dropdown(
                        id='dropdown-sankey',
                        options=[{'label': 'Toutes les écoles', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sorted(df['Ecole'].unique())],
                        value='ALL', clearable=False
                    ),
                    html.Button("Réinitialiser la vue Polytech", id="btn-reset-vf", n_clicks=0, style={'marginTop': '10px', 'width': '100%'})
                ]),
                html.Div(id='sankey-headers', style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '0 40px', 'marginBottom': '10px', 'borderBottom': '2px solid #f0f0f0'}),
                dcc.Graph(id='sankey-graph')
            ]),
            html.Div(style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '15px'}, children=[
            html.H3("Analyse de Corrélation Globale", style={'textAlign': 'center'}),
            dcc.Graph(id='competence-metier-scatter', figure=px.scatter(df_filtered, x='Nombre de Compétences', y='Nombre de Métiers', color='Ecole', hover_name='Formation',size='Nombre de Compétences', template="plotly_white"))
        ])
        ]),
        ##########fin polytech reseau######

        # ═══════════════════════════════════════════════════════════════
        #                MAIN LAYOUT avec sidebar fixe (Learnagement)
        # ═══════════════════════════════════════════════════════════════
        html.Div(
            [
                # Sidebar fixe à gauche
                html.Div(
                    create_filter_section(),
                    className="app-sidebar",
                ),

                # Contenu principal
                html.Div(
                    [
                        # KPIs
                        html.Div(
                            [
                                create_section_header(
                                    "Vue d'ensemble de Polytech Annecy",
                                    "Indicateurs clés de votre curriculum",
                                ),
                                html.Div(id="stats-dashboard", className="stats-grid"),
                            ],
                            className="section",
                        ),

                        # Analyses principales
                        html.Div(
                            [
                                create_section_header(
                                    "Analyses principales",
                                    "Statistiques par compétences",
                                ),
                                # Ligne 1 : Les deux bar charts côte à côte
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz5-critical",
                                                    config={"displayModeBar": True, "displaylogo": False},
                                                )
                                            ],
                                            className="card viz-card",
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz3-heatmap",
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

                        # Mon parcours
                        html.Div(
                            [
                                create_section_header(
                                    "Mon parcours",
                                    "Exploration interactive par niveaux",
                                ),
                                html.Div(
                                    id="breadcrumb-nav",
                                    className="breadcrumb",
                                ),
                                html.Div(
                                    [
                                        html.Button(
                                            "🏠 Vue globale",
                                            id="btn-global",
                                            n_clicks=0,
                                            className="btn btn-primary",
                                        ),
                                        html.Button(
                                            "← Retour",
                                            id="btn-back",
                                            n_clicks=0,
                                            className="btn btn-outline",
                                        ),
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
                                create_section_header(
                                    "Autres analyses",
                                    "Radar et flux",
                                ),
                                # Ligne 2 : Radar chart et graphe vide
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
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="viz4-flow",
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
                html.P(
                    "Utilisez les filtres pour adapter le dashboard à votre parcours",
                    className="footer-text",
                ),
                html.P(
                    "Learnagement-Approche par compétences, 2025",
                    className="footer-credit",
                ),
            ],
            className="app-footer",
        ),
    ],
    className="app-wrapper",
)

# --- CALLBACKS POLYTECH ---

@app.callback(Output('dropdown-sankey', 'value'), [Input({'type': 'marker-ecole', 'index': ALL}, 'n_clicks'), Input('btn-reset-vf', 'n_clicks')], prevent_initial_call=True)
def sync_selection_vf(n_clicks_list, reset_clicks):
    ctx = callback_context
    if "btn-reset-vf" in ctx.triggered[0]['prop_id']: return "ALL"
    return json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']

@app.callback(Output("layer-markers", "children"), Input("layer-markers", "id"))
def render_markers_vf(_):
    return [dl.CircleMarker(center=coords, id={'type': 'marker-ecole', 'index': school}, radius=10, color=COLORS['primary'], children=[dl.Tooltip(school)]) for school, coords in coords_ecoles.items()]

@app.callback([Output('graphique-detail', 'figure'), Output('titre-ecole', 'children'), Output('welcome-message', 'style')], [Input('dropdown-sankey', 'value')])
def update_ui_bar_vf(selected_school):
    if selected_school == "ALL": return go.Figure(), "🏫 Sélectionnez une école", {'display': 'block'}
    df_school = df_filtered[df_filtered['Ecole'] == selected_school].sort_values('Nombre de Compétences')
    fig = px.bar(df_school, y='Formation', x='Nombre de Compétences', orientation='h', color='Nombre de Compétences', color_continuous_scale='Viridis', template='plotly_white')
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
                if isinstance(item, list): item = item[0]
                if isinstance(item, str) and len(item) > 3: debouches.append(reparer_texte(item.split(',')[0].split('(')[0].strip()[:35]))
    
    formations, debouches = sorted(list(set(formations))), sorted(list(set(debouches)))
    all_labels = formations + ecoles + debouches
    label_map = {name: i for i, name in enumerate(all_labels)}
    display_labels = [n if n not in debouches else "" for n in all_labels]
    for school, forms in s_data.items():
        for f in forms:
            f_idx = label_map[reparer_texte(f.get('formation', '').replace('-', ' ').title())]; e_idx = label_map[school]
            sources.append(f_idx); targets.append(e_idx); values.append(1)
            combined = (f.get('secteurs', []) or []) + (f.get('metiers', []) or [])
            for item in combined:
                if isinstance(item, list): item = item[0]
                if isinstance(item, str) and len(item) > 3:
                    clean = reparer_texte(item.split(',')[0].split('(')[0].strip()[:35])
                    sources.append(e_idx); targets.append(label_map[clean]); values.append(1)

    fig = go.Figure(data=[go.Sankey(node=dict(pad=50, thickness=20, label=display_labels, customdata=all_labels, hovertemplate='%{customdata}<extra></extra>', color="royalblue"), link=dict(source=sources, target=targets, value=values, color="rgba(100, 150, 250, 0.2)"))])
    fig.update_layout(font_size=12, height=700, margin=dict(t=20, b=20, l=200, r=10))
    headers = [html.Div("FILIÈRE", style={'fontWeight': 'bold', 'width': '33%'}), html.Div("ÉCOLE", style={'fontWeight': 'bold', 'textAlign': 'center', 'width': '33%'}), html.Div("DÉBOUCHÉS", style={'fontWeight': 'bold', 'textAlign': 'right', 'width': '33%'})]
    return fig, headers

@app.callback(Output('liste-competences', 'children'), [Input('graphique-detail', 'clickData'), Input('sankey-graph', 'clickData')], [State('sankey-graph', 'figure')])
def unified_click_handler_vf(barData, sankeyData, sankeyFig):
    ctx = callback_context
    if not ctx.triggered: return html.Em("Cliquez sur un élément pour voir le détail.")
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if input_id == 'graphique-detail' and barData:
        f_nom = barData['points'][0]['y']
        for s in datavf:
            for f in datavf[s]:
                if reparer_texte(f['formation'].replace('-', ' ').title()) == f_nom:
                    return html.Div([html.B(f"Compétences pour {f_nom} :"), html.Ul([html.Li(reparer_texte(c)) for c in f.get('competences', [])])])
    return "Aucun détail trouvé."


# --- CALLBACKS LEARNAGEMENT ---
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

    trigger_id = (
        ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "none"
    )

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
                # Retrouver l'ID du module à partir du nom
                clicked_module = module_ids.get(clicked_module_name, clicked_module_name)
                # Si c'est toujours une string, essayer de parser
                if isinstance(clicked_module, str) and clicked_module.replace('.', '').isdigit():
                    clicked_module = float(clicked_module)
                new_level = "module"
                new_module = clicked_module

    try:
        df_pivot_global, df_pivot_semester, df_pivot_module = compute_competency_counts_per_module(
            filtered_df
        )
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erreur: {str(e)}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig, "global", None, None, html.Div("Erreur")

    if new_level == "global":
        fig = create_heatmap_for_global(df_pivot_global)
        breadcrumb = html.Span("Vue globale", className="breadcrumb-item active")
    elif new_level == "semester" and new_semester is not None:
        available_semesters = (
            df_pivot_semester.index.get_level_values("niveau").unique().tolist()
        )
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
        available_modules = (
            df_pivot_module.index.get_level_values("id_module").unique().tolist()
        )
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

@app.callback(
    Output("stats-dashboard", "children"),
    Input("filtered-data-store", "data"),
)
def update_stats(filtered_data_json):
    df = df_main
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz7_statistics_dashboard(filtered_df)

@app.callback(
    Output("viz2-radar", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz2(filtered_data_json):
    df = df_main
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz2_workload_radar(filtered_df)

@app.callback(
    Output("viz3-heatmap", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz3(filtered_data_json):
    df = df_main
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz3_competency_heatmap(filtered_df)

    df = df_main
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz4_learning_flow(filtered_df)

@app.callback(
    Output("viz5-critical", "figure"),
    Input("filtered-data-store", "data"),
)
def update_viz5(filtered_data_json):
    df = df_main
    if filtered_data_json is None:
        filtered_df = df.copy()
    else:
        filtered_df = pd.read_json(StringIO(filtered_data_json), orient="split")
        filtered_df["niveau_code"] = filtered_df.apply(parse_competencies, axis=1)
    return viz5_critical_competencies(filtered_df)


# ═══════════════════════════════════════════════════════════════════
#                          LANCEMENT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
