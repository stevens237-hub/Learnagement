import dash
from dash import dcc, html, Output, Input, State, ALL, callback_context
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import dash_leaflet as dl

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
        data = json.load(f)
except FileNotFoundError:
    print(f"Erreur: Le fichier {file_name} est introuvable.")
    data = {}

coords_ecoles = {
    "Polytech Nantes": [47.282, -1.520], "Polytech Montpellier": [43.632, 3.863],
    "Polytech Annecy": [45.920, 6.138], "Polytech Paris saclay": [48.706, 2.169],
    "Polytech Tours": [47.354, 0.704], "Polytech Nice Sophia": [43.616, 7.072],
    "Polytech Angers": [47.481, -0.594], "Polytech Clermont": [45.758, 3.111],
    "Polytech Grenoble": [45.193, 5.767], "Polytech Lyon": [45.783, 4.868],
    "Polytech Nancy": [48.665, 6.155]
}

records = []
for school, formations in data.items():
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

# --- 3. MISE EN PLACE DU DASHBOARD ---

app = dash.Dash(__name__, external_stylesheets=['https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap'])

COLORS = {"primary": "#0056b3", "background": "#f4f7f6", "card": "#ffffff", "text": "#2d3436"}

app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'fontFamily': '"Poppins", sans-serif', 'padding': '0', 'margin': '0'}, children=[
    
    html.Header(style={'backgroundColor': COLORS['primary'], 'padding': '20px', 'color': 'white', 'textAlign': 'center', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'}, children=[
        html.H1('Réseau Polytech : Analyse des Compétences', style={'margin': '0', 'fontWeight': '600'}),
    ]),
    
    html.Div(style={'padding': '20px'}, children=[
        html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start'}, children=[
            html.Div(style={'flex': '1.5'}, children=[
                html.H3("🌍 Localisation des écoles"),
                dl.Map([dl.TileLayer(), dl.LayerGroup(id="layer-markers")], center=[46.5, 2.5], zoom=6, style={'height': '65vh', 'borderRadius': '12px'})
            ]),
            html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'height': '65vh'}, children=[
                html.H3(id='titre-ecole', children="🏫 Sélectionnez une école", style={'marginTop': '0', 'marginBottom': '10px'}),
                html.Div(style={'flex': '1', 'backgroundColor': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.05)', 'padding': '15px', 'overflowY': 'auto'}, children=[
                    html.Div(id='welcome-message', style={'textAlign': 'center', 'paddingTop': '50px'}, children=[html.P("Cliquez sur un point bleu ou une barre pour voir les détails.")]),
                    dcc.Graph(id='graphique-detail', style={'height': '350px'}, config={'displayModeBar': False}),
                    # Zone de texte pour afficher les résultats des clics
                    html.Div(id='liste-competences', style={'marginTop': '20px', 'fontSize': '14px', 'borderTop': '1px solid #eee', 'paddingTop': '10px'})
                ])
            ])
        ]),

        html.Div(style={'backgroundColor': 'white', 'padding': '25px', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)', 'marginBottom': '30px', 'marginTop': '30px'}, children=[
            html.H3("🧭 Flux de carrière (Cliquez sur les barres pour le détail)", style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.Div(style={'width': '50%', 'margin': '0 auto 30px auto'}, children=[
                dcc.Dropdown(
                    id='dropdown-sankey',
                    options=[{'label': 'Toutes les écoles', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sorted(df['Ecole'].unique())],
                    value='ALL', clearable=False
                ),
                html.Button("Réinitialiser la vue", id="btn-reset", n_clicks=0, style={'marginTop': '10px', 'width': '100%'})
            ]),

            html.Div(id='sankey-headers', style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '0 40px', 'marginBottom': '10px', 'borderBottom': '2px solid #f0f0f0'}),
            dcc.Graph(id='sankey-graph')
        ]),

        html.Div(style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '15px'}, children=[
            html.H3("Analyse de Corrélation Globale", style={'textAlign': 'center'}),
            dcc.Graph(id='competence-metier-scatter', figure=px.scatter(df_filtered, x='Nombre de Compétences', y='Nombre de Métiers', color='Ecole', hover_name='Formation', template="plotly_white"))
        ])
    ])
])

# --- 4. CALLBACKS ---

@app.callback(Output('dropdown-sankey', 'value'), [Input({'type': 'marker-ecole', 'index': ALL}, 'n_clicks'), Input('btn-reset', 'n_clicks')], prevent_initial_call=True)
def sync_selection(n_clicks_list, reset_clicks):
    ctx = callback_context
    if "btn-reset" in ctx.triggered[0]['prop_id']: return "ALL"
    return json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']

@app.callback(Output("layer-markers", "children"), Input("layer-markers", "id"))
def render_markers(_):
    return [dl.CircleMarker(center=coords, id={'type': 'marker-ecole', 'index': school}, radius=10, color=COLORS['primary'], children=[dl.Tooltip(school)]) for school, coords in coords_ecoles.items()]

@app.callback([Output('graphique-detail', 'figure'), Output('titre-ecole', 'children'), Output('welcome-message', 'style')], [Input('dropdown-sankey', 'value')])
def update_ui_bar(selected_school):
    if selected_school == "ALL": return go.Figure(), "🏫 Sélectionnez une école", {'display': 'block'}
    df_school = df_filtered[df_filtered['Ecole'] == selected_school].sort_values('Nombre de Compétences')
    fig = px.bar(df_school, y='Formation', x='Nombre de Compétences', orientation='h', color='Nombre de Compétences', color_continuous_scale='Viridis', template='plotly_white')
    fig.update_layout(margin=dict(l=150, r=20, t=20, b=40), xaxis=dict(dtick=1), coloraxis_showscale=False)
    return fig, selected_school, {'display': 'none'}

@app.callback([Output('sankey-graph', 'figure'), Output('sankey-headers', 'children')], Input('dropdown-sankey', 'value'))
def update_sankey(selected_school):
    s_data = data if selected_school == 'ALL' else {selected_school: data.get(selected_school, [])}
    sources, targets, values, labels = [], [], [], []
    formations, ecoles, debouches = [], list(s_data.keys()), []

    for school, forms in s_data.items():
        for f in forms:
            f_name = reparer_texte(f.get('formation', '').replace('-', ' ').title())
            formations.append(f_name)
            combined = (f.get('secteurs', []) or []) + (f.get('metiers', []) or [])
            for item in combined:
                if isinstance(item, list): item = item[0]
                if isinstance(item, str) and len(item) > 3:
                    clean = reparer_texte(item.split(',')[0].split('(')[0].strip()[:35])
                    debouches.append(clean)
    
    formations, debouches = sorted(list(set(formations))), sorted(list(set(debouches)))
    all_labels = formations + ecoles + debouches
    label_map = {name: i for i, name in enumerate(all_labels)}
    
    # On laisse les labels vides sur le graphique pour la 3eme colonne mais on garde customdata pour le clic/survol
    display_labels = [n if n not in debouches else "" for n in all_labels]

    for school, forms in s_data.items():
        for f in forms:
            f_idx = label_map[reparer_texte(f.get('formation', '').replace('-', ' ').title())]
            e_idx = label_map[school]
            sources.append(f_idx); targets.append(e_idx); values.append(1)
            combined = (f.get('secteurs', []) or []) + (f.get('metiers', []) or [])
            for item in combined:
                if isinstance(item, list): item = item[0]
                if isinstance(item, str) and len(item) > 3:
                    clean = reparer_texte(item.split(',')[0].split('(')[0].strip()[:35])
                    sources.append(e_idx); targets.append(label_map[clean]); values.append(1)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=50, thickness=20, line=dict(color="black", width=0.5), 
            label=display_labels,
            customdata=all_labels, # On stocke les vrais noms ici
            hovertemplate='%{customdata}<extra></extra>', # Affiche le nom au survol
            color="royalblue"
        ),
        link=dict(source=sources, target=targets, value=values, color="rgba(100, 150, 250, 0.2)")
    )])
    fig.update_layout(font_size=12, height=1200, margin=dict(t=20, b=20, l=200, r=10))

    header_right = "DÉBOUCHÉS" if debouches else ""
    headers = [
        html.Div("FILIÈRE", style={'fontWeight': 'bold', 'width': '33%'}),
        html.Div("ÉCOLE", style={'fontWeight': 'bold', 'textAlign': 'center', 'width': '33%'}),
        html.Div(header_right, style={'fontWeight': 'bold', 'textAlign': 'right', 'width': '33%'})
    ]
    return fig, headers

# CALLBACK UNIQUE POUR LES CLICS (Bar Chart + Sankey)
@app.callback(
    Output('liste-competences', 'children'),
    [Input('graphique-detail', 'clickData'),
     Input('sankey-graph', 'clickData')]
)
def unified_click_handler(barData, sankeyData):
    ctx = callback_context
    if not ctx.triggered: return html.Em("Cliquez sur un élément pour voir le détail.")
    
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # CAS 1 : CLIC SUR LE BAR CHART (Compétences)
    if input_id == 'graphique-detail' and barData:
        f_nom = barData['points'][0]['y']
        for s in data:
            for f in data[s]:
                if reparer_texte(f['formation'].replace('-', ' ').title()) == f_nom:
                    return html.Div([
                        html.B(f"Compétences pour {f_nom} :"),
                        html.Ul([html.Li(reparer_texte(c)) for c in f.get('competences', [])])
                    ])

    # CAS 2 : CLIC SUR LE SANKEY (Métiers ou Formations)
    if input_id == 'sankey-graph' and sankeyData:
        # Récupérer le nom du nœud cliqué via customdata
        node_idx = sankeyData['points'][0]['pointNumber']
        # On doit retrouver le nom dans la liste globale
        # Pour faire simple, on récupère le label via customdata (si disponible) ou on le déduit
        node_name = sankeyData['points'][0].get('label', '') 
        # Si le label était vide (cas des métiers), on le récupère autrement
        if node_name == "":
            # On cherche dans les données du graphique actuel
            node_name = ctx.states.get('sankey-graph.figure', {}).get('data', [{}])[0].get('node', {}).get('customdata', [])[node_idx]

        # Si c'est une formation -> montrer les compétences
        for s in data:
            for f in data[s]:
                if reparer_texte(f['formation'].replace('-', ' ').title()) == node_name:
                    return html.Div([
                        html.B(f"Compétences pour {node_name} :"),
                        html.Ul([html.Li(reparer_texte(c)) for c in f.get('competences', [])])
                    ])
        
        # Si c'est un métier/secteur -> montrer les formations liées
        formations_liees = []
        for s in data:
            for f in data[s]:
                comb = (f.get('secteurs', []) or []) + (f.get('metiers', []) or [])
                for item in comb:
                    if isinstance(item, list): item = item[0]
                    if isinstance(item, str) and node_name in reparer_texte(item):
                        formations_liees.append(reparer_texte(f['formation'].replace('-', ' ').title()))
        
        if formations_liees:
            return html.Div([
                html.B(f"Débouché : {node_name}"),
                html.P("Formations correspondantes :"),
                html.Ul([html.Li(f) for f in set(formations_liees)])
            ])

    return "Aucun détail trouvé."

if __name__ == '__main__':
    app.run(debug=True)