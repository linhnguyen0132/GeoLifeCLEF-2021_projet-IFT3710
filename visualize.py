import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 1. Charger les résultats de ton IA
chemin_csv = "C:/Users/abdou/OneDrive/Documents/Python projects/3710/GLC_Project/Asset/rapport_habitats_critiques.csv"
print(f"Chargement des alertes depuis {chemin_csv}...")
df = pd.read_csv(chemin_csv)

# --- SÉCURITÉS ANTI-PLANTAGE ---
# Sécurité 1 : On supprime les lignes qui auraient des coordonnées GPS corrompues ou vides
df = df.dropna(subset=['latitude', 'longitude'])

# Sécurité 2 : On trie par probabilité décroissante et on garde seulement le Top 1000
df = df.sort_values(by="probabilite_modele", ascending=False).head(1000)
# -------------------------------

# 2. Créer la carte de base
carte = folium.Map(location=[46.2276, 2.2137], zoom_start=6, tiles="cartodb positron")

# 3. Créer le groupe "Cluster" pour éviter de saturer la carte
marker_cluster = MarkerCluster().add_to(carte)

# 4. Placer les points
print(f"Dessin des {len(df)} points les plus critiques sur la carte en cours...")
for index, row in df.iterrows():
    texte_popup = f"""
    <b>Espèce :</b> {row['nom_espece']}<br>
    <b>ID :</b> {row['species_id']}<br>
    <b>Probabilité IA :</b> {row['probabilite_modele']}%<br>
    <b>Rang Top-30 :</b> {row['rang_top30']}
    """
    
    # On ajoute le point dans le CLUSTER, pas directement sur la carte
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(texte_popup, max_width=300),
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(marker_cluster)

# 5. Sauvegarder
nom_fichier_sortie = "carte_alertes_iucn.html"
carte.save(nom_fichier_sortie)
print(f"SUCCÈS ! La carte a été générée sous le nom : {nom_fichier_sortie}")