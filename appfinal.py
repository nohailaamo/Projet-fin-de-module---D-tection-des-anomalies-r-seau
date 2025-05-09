import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator
import time

# Configuration de la page
st.set_page_config(page_title="Détection d'anomalies Réseau",page_icon="applogo.png", layout="wide")
st.title("Détection d'anomalies Réseau")

# Sidebar pour les configurations
st.sidebar.title("⚙️ Configuration")

# Section upload
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV pour analyse", type=['csv'])

if uploaded_file is not None:
    # Chargement des données avec feedback utilisateur
    with st.spinner('Chargement des données...'):
        try:
            # Essayer d'abord avec header
            df = pd.read_csv(uploaded_file)
            # Vérifier si les noms de colonnes semblent être des données
            if df.columns.str.contains(r'^\d+$').all():
                df = pd.read_csv(uploaded_file, header=None)
        except:
            # Si échec, essayer sans header
            df = pd.read_csv(uploaded_file, header=None)
        
        # Création d'une copie des données originales
        df_original = df.copy()
    
    # Affichage des informations sur le dataset
    st.subheader("📋 Informations sur le dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nombre d'observations", df.shape[0])
        st.metric("Nombre de caractéristiques", df.shape[1])
    with col2:
        missing_values = df.isnull().sum().sum()
        st.metric("Valeurs manquantes", missing_values)
        if missing_values > 0:
            st.warning(f"⚠️ Le dataset contient {missing_values} valeurs manquantes.")
    
    # Aperçu des données
    with st.expander("🔎 Aperçu des données brutes"):
        st.write(df.head(10))
        
        # Informations sur les types de données
        st.subheader("Types de données")
        dtypes_df = pd.DataFrame(df.dtypes, columns=['Type de données'])
        st.write(dtypes_df)

    # Prétraitement des données
    st.subheader("🔧 Prétraitement des données")
    
    # Sélection des colonnes
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Interface pour sélectionner les colonnes à utiliser
    with st.expander("Sélection des caractéristiques"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Caractéristiques numériques:")
            selected_numeric = st.multiselect(
                "Sélectionnez les caractéristiques numériques à utiliser",
                numeric_cols,
                default=numeric_cols
            )
        
        with col2:
            st.write("Caractéristiques catégorielles:")
            selected_categorical = st.multiselect(
                "Sélectionnez les caractéristiques catégorielles à utiliser",
                categorical_cols,
                default=categorical_cols
            )
    
    # Traitement des valeurs manquantes
    handle_missing = st.radio(
        "Traitement des valeurs manquantes:",
        ("Supprimer les lignes avec valeurs manquantes", "Remplacer par la moyenne/mode")
    )
    
    with st.spinner('Prétraitement en cours...'):
        # Application du traitement des valeurs manquantes
        if handle_missing == "Supprimer les lignes avec valeurs manquantes":
            df_clean = df[selected_numeric + selected_categorical].dropna()
        else:
            df_clean = df[selected_numeric + selected_categorical].copy()
            # Remplacement des valeurs manquantes numériques par la moyenne
            for col in selected_numeric:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            # Remplacement des valeurs manquantes catégorielles par le mode
            for col in selected_categorical:
                if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Vérification après prétraitement
        if df_clean.empty:
            st.error("❌ Après prétraitement, le dataset est vide. Veuillez revoir vos paramètres.")
            st.stop()
        
        # Préparation des données pour le clustering
        if len(selected_numeric) > 0 or len(selected_categorical) > 0:
            # Création d'un préprocesseur
            preprocessor_steps = []
            
            if selected_numeric:
                preprocessor_steps.append(('num', StandardScaler(), selected_numeric))
            
            if selected_categorical:
                preprocessor_steps.append(('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), selected_categorical))
            
            if preprocessor_steps:
                preprocessor = ColumnTransformer(preprocessor_steps)
                X_preprocessed = preprocessor.fit_transform(df_clean)
                
                # Création d'un DataFrame des données prétraitées pour l'affichage
                if len(selected_categorical) > 0:
                    # Récupération des noms des colonnes après one-hot encoding
                    cat_features = []
                    for i, category in enumerate(selected_categorical):
                        cat_encoder = preprocessor.transformers_[1][1]
                        categories = cat_encoder.categories_[i]
                        for cat in categories:
                            cat_features.append(f"{category}_{cat}")
                    
                    feature_names = selected_numeric + cat_features
                else:
                    feature_names = selected_numeric
                
                # Assurer que nous avons le bon nombre de noms de colonnes
                if len(feature_names) != X_preprocessed.shape[1]:
                    feature_names = [f"feature_{i}" for i in range(X_preprocessed.shape[1])]
                
                df_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
            else:
                st.error("❌ Veuillez sélectionner au moins une caractéristique.")
                st.stop()
        else:
            st.error("❌ Aucune caractéristique sélectionnée. Veuillez en choisir au moins une.")
            st.stop()
    
    # Affichage des statistiques descriptives
    with st.expander("📊 Statistiques descriptives"):
        if len(selected_numeric) > 0:
            st.write(df_clean[selected_numeric].describe())
        
        # Matrice de corrélation pour les variables numériques
        if len(selected_numeric) > 1:
            st.subheader("Matrice de corrélation")
            corr_matrix = df_clean[selected_numeric].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                        square=True, linewidths=.5, annot=True, fmt=".2f")
            st.pyplot(fig)
    
    # Analyse en Composantes Principales
    with st.expander("🧩 Analyse en Composantes Principales (PCA)"):
        n_components = min(5, X_preprocessed.shape[1])
        n_components = st.slider("Nombre de composantes principales", 2, n_components, 2)
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_preprocessed)
        
        # Variance expliquée
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Graphique de la variance expliquée
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='skyblue',
                label='Variance individuelle')
        ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', 
                label='Variance cumulée')
        ax.set_xlabel('Composantes principales')
        ax.set_ylabel('Ratio de variance expliquée')
        ax.set_title('Variance expliquée par composante principale')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Affichage des 2 premières composantes
        st.subheader("Visualisation des 2 premières composantes principales")
        pca_df = pd.DataFrame(data=pca_result[:, :2], columns=['PC1', 'PC2'])
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', opacity=0.7,
                        labels={'PC1': f'PC1 ({explained_variance[0]:.2%})',
                                'PC2': f'PC2 ({explained_variance[1]:.2%})'},
                        title='Projection PCA des données')
        st.plotly_chart(fig, use_container_width=True)
    
    # Algorithmes de clustering
    st.subheader("🧠 Algorithmes de clustering")
    
    # Sélection de l'algorithme
    algo_choice = st.radio(
        "Choisissez les algorithmes à exécuter:",
        ("K-Means", "DBSCAN", "Isolation Forest", "Tous")
    )
    
    # K-Means
    if algo_choice in ["K-Means", "Tous"]:
        st.markdown("### K-Means")
        
        # Détermination du nombre optimal de clusters (méthode du coude)
        with st.expander("🔍 Trouver le nombre optimal de clusters"):
            max_clusters = min(15, X_preprocessed.shape[0] // 5)  # Limite raisonnable
            
            if st.button("Calculer la méthode du coude"):
                with st.spinner("Calcul en cours..."):
                    inertia = []
                    K_range = range(1, max_clusters + 1)
                    
                    for k in K_range:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                        kmeans.fit(X_preprocessed)
                        inertia.append(kmeans.inertia_)
                    
                    # Méthode du coude
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(K_range, inertia, 'bo-')
                    ax.set_xlabel('Nombre de clusters')
                    ax.set_ylabel('Inertie')
                    ax.set_title('Méthode du coude pour déterminer le nombre optimal de clusters')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Détection automatique du coude
                    try:
                        kl = KneeLocator(K_range, inertia, curve='convex', direction='decreasing')
                        optimal_k = kl.elbow
                        ax.axvline(x=optimal_k, color='r', linestyle='--', 
                                  label=f'Coude détecté à k={optimal_k}')
                        ax.legend()
                    except:
                        st.warning("⚠️ Impossible de détecter automatiquement le coude.")
                    
                    st.pyplot(fig)
        
        # Paramètres K-Means
        n_clusters = st.slider("Nombre de clusters pour K-Means", 2, 10, 3)
        
        # Exécution de K-Means
        with st.spinner("Exécution de K-Means..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans_labels = kmeans.fit_predict(X_preprocessed)
            
            # Ajout des labels au DataFrame
            df_clean['KMeans_Cluster'] = kmeans_labels
            
            # Métriques d'évaluation
            if n_clusters > 1 and n_clusters < len(X_preprocessed):
                silhouette_avg = silhouette_score(X_preprocessed, kmeans_labels)
                davies_bouldin = davies_bouldin_score(X_preprocessed, kmeans_labels)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score Silhouette", f"{silhouette_avg:.3f}",
                             delta="+0.1" if silhouette_avg > 0.5 else "-0.1", 
                             delta_color="normal")
                with col2:
                    st.metric("Score Davies-Bouldin", f"{davies_bouldin:.3f}",
                             delta="-0.1" if davies_bouldin < 1.0 else "+0.1", 
                             delta_color="inverse")
            
            # Visualisation PCA avec clusters
            pca_vis = PCA(n_components=2)
            pca_result = pca_vis.fit_transform(X_preprocessed)
            
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = kmeans_labels
            
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                            color_continuous_scale=px.colors.qualitative.G10,
                            labels={'PC1': 'Composante Principale 1', 
                                    'PC2': 'Composante Principale 2'},
                            title='Visualisation des clusters K-Means (PCA)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des clusters
            st.subheader("Analyse des clusters K-Means")
            
            # Distribution des observations par cluster
            cluster_counts = pd.DataFrame(df_clean['KMeans_Cluster'].value_counts()).reset_index()
            cluster_counts.columns = ['Cluster', 'Nombre d\'observations']
            
            fig = px.bar(cluster_counts, x='Cluster', y='Nombre d\'observations',
                        title='Distribution des observations par cluster')
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques des clusters
            with st.expander("📊 Statistiques détaillées par cluster"):
                selected_cluster = st.selectbox(
                    "Sélectionnez un cluster pour voir ses statistiques",
                    sorted(df_clean['KMeans_Cluster'].unique())
                )
                
                if selected_numeric:
                    st.write(df_clean[df_clean['KMeans_Cluster'] == selected_cluster][selected_numeric].describe())
                    
                    # Boxplots des caractéristiques numériques
                    if len(selected_numeric) > 0:
                        fig = px.box(df_clean[df_clean['KMeans_Cluster'] == selected_cluster],
                                    y=selected_numeric, 
                                    title=f'Distribution des caractéristiques pour le cluster {selected_cluster}')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Caractéristiques catégorielles
                if selected_categorical:
                    for cat in selected_categorical:
                        if df_clean[cat].dtype == 'object' or df_clean[cat].dtype.name == 'category':
                            cat_dist = df_clean[df_clean['KMeans_Cluster'] == selected_cluster][cat].value_counts()
                            
                            fig = px.pie(values=cat_dist.values, names=cat_dist.index,
                                        title=f'Distribution de {cat} dans le cluster {selected_cluster}')
                            st.plotly_chart(fig, use_container_width=True)
            
            # Comparaison des centroïdes
            with st.expander("📍 Centroïdes des clusters"):
                # Récupération des centroïdes
                centroids = pd.DataFrame(kmeans.cluster_centers_, columns=df_preprocessed.columns)
                
                # Heatmap des centroïdes
                fig = px.imshow(centroids, text_auto=True,
                                labels=dict(x="Caractéristique", y="Cluster", color="Valeur"),
                                title="Heatmap des centroïdes des clusters")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des centroïdes
                st.write(centroids)
    
    # DBSCAN
    if algo_choice in ["DBSCAN", "Tous"]:
        st.markdown("### DBSCAN")
        
        # Paramètres DBSCAN
        col1, col2 = st.columns(2)
        with col1:
            eps_value = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        with col2:
            min_samples = st.slider("Minimum d'échantillons", 2, 20, 5)
        
        # Exécution de DBSCAN
        with st.spinner("Exécution de DBSCAN..."):
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(X_preprocessed)
            
            # Ajout des labels au DataFrame
            df_clean['DBSCAN_Cluster'] = dbscan_labels
            
            # Métriques
            n_clusters_ = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            n_noise_ = list(dbscan_labels).count(-1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre de clusters identifiés", n_clusters_)
            with col2:
                noise_percent = n_noise_ / len(dbscan_labels) * 100
                st.metric("Points de bruit", f"{n_noise_} ({noise_percent:.1f}%)")
            
            # Visualisation PCA avec clusters DBSCAN
            pca_vis = PCA(n_components=2)
            pca_result = pca_vis.fit_transform(X_preprocessed)
            
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = ["Bruit" if l == -1 else f"Cluster {l}" for l in dbscan_labels]
            
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                            labels={'PC1': 'Composante Principale 1', 
                                    'PC2': 'Composante Principale 2'},
                            title='Visualisation des clusters DBSCAN (PCA)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des clusters DBSCAN
            st.subheader("Analyse des clusters DBSCAN")
            
            # Distribution des observations par cluster
            cluster_counts = pd.DataFrame(df_clean['DBSCAN_Cluster'].value_counts()).reset_index()
            cluster_counts.columns = ['Cluster', 'Nombre d\'observations']
            
            fig = px.bar(cluster_counts, x='Cluster', y='Nombre d\'observations',
                        title='Distribution des observations par cluster')
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparaison de t-SNE
            with st.expander("🔄 Visualisation t-SNE"):
                st.info("t-SNE est particulièrement utile pour visualiser des données à haute dimension")
                
                with st.spinner("Calcul de t-SNE en cours..."):
                    # Application de t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_preprocessed)-1))
                    tsne_results = tsne.fit_transform(X_preprocessed)
                    
                    # Création du DataFrame pour la visualisation
                    tsne_df = pd.DataFrame(data=tsne_results, columns=['t-SNE1', 't-SNE2'])
                    tsne_df['Cluster'] = ["Bruit" if l == -1 else f"Cluster {l}" for l in dbscan_labels]
                    
                    # Création du graphique
                    fig = px.scatter(tsne_df, x='t-SNE1', y='t-SNE2', color='Cluster',
                                    title='Visualisation t-SNE des clusters DBSCAN')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Isolation Forest
    if algo_choice in ["Isolation Forest", "Tous"]:
        st.markdown("### Isolation Forest")
        
        # Paramètres Isolation Forest
        contamination = st.slider("Taux de contamination attendu", 0.01, 0.5, 0.1, 0.01)
        
        # Exécution de Isolation Forest
        with st.spinner("Exécution de Isolation Forest..."):
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            iso_forest_pred = iso_forest.fit_predict(X_preprocessed)
            
            # Conversion des prédictions (-1 pour anomalie, 1 pour normal) en labels binaires
            iso_forest_labels = np.where(iso_forest_pred == -1, 1, 0)  # 1 pour anomalie, 0 pour normal
            
            # Ajout des labels au DataFrame
            df_clean['IsAnomaly'] = iso_forest_labels
            
            # Calcul des scores d'anomalie
            anomaly_scores = -iso_forest.score_samples(X_preprocessed)
            df_clean['AnomalyScore'] = anomaly_scores
            
            # Statistiques sur les anomalies
            n_anomalies = np.sum(iso_forest_labels)
            anomaly_percent = n_anomalies / len(iso_forest_labels) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre d'anomalies détectées", n_anomalies)
            with col2:
                st.metric("Pourcentage d'anomalies", f"{anomaly_percent:.1f}%")
            
            # Visualisation des anomalies
            pca_vis = PCA(n_components=2)
            pca_result = pca_vis.fit_transform(X_preprocessed)
            
            pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
            pca_df['Anomalie'] = ["Anomalie" if l == 1 else "Normal" for l in iso_forest_labels]
            pca_df['Score d\'anomalie'] = anomaly_scores
            
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Anomalie',
                            color_discrete_map={'Normal': 'blue', 'Anomalie': 'red'},
                            labels={'PC1': 'Composante Principale 1', 
                                    'PC2': 'Composante Principale 2'},
                            title='Visualisation des anomalies (Isolation Forest)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution des scores d'anomalie
            fig = px.histogram(pca_df, x='Score d\'anomalie', color='Anomalie',
                              title='Distribution des scores d\'anomalie',
                              color_discrete_map={'Normal': 'blue', 'Anomalie': 'red'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Top des anomalies
            st.subheader("Top 10 des anomalies les plus significatives")
            top_anomalies = df_clean[df_clean['IsAnomaly'] == 1].sort_values(by='AnomalyScore', ascending=False).head(10)
            st.write(top_anomalies)
    
    # Comparaison des algorithmes
    if algo_choice == "Tous":
        st.subheader("📊 Comparaison des algorithmes")
        
        # Création d'un DataFrame de résultats
        results_df = df_clean.copy()
        
        # Tableau de contingence K-Means vs DBSCAN
        st.write("K-Means vs DBSCAN")
        contingency = pd.crosstab(results_df['KMeans_Cluster'], results_df['DBSCAN_Cluster'])
        st.write(contingency)
        
        # Visualisation de la comparaison
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                        symbol='Anomalie', symbol_map={'Normal': 'circle', 'Anomalie': 'x'},
                        title='Comparaison des résultats des différents algorithmes')
        st.plotly_chart(fig, use_container_width=True)
    
    # Export des résultats
    st.subheader("📥 Exportation des résultats")
    
    # Préparation des données pour l'export
    export_df = df_original.copy()
    
    if 'KMeans_Cluster' in df_clean.columns:
        # Aligner les indices
        export_df = export_df.loc[df_clean.index]
        export_df['KMeans_Cluster'] = df_clean['KMeans_Cluster']
    
    if 'DBSCAN_Cluster' in df_clean.columns:
        if 'KMeans_Cluster' not in df_clean.columns:
            # Aligner les indices si pas déjà fait
            export_df = export_df.loc[df_clean.index]
        export_df['DBSCAN_Cluster'] = df_clean['DBSCAN_Cluster']
    
    if 'IsAnomaly' in df_clean.columns:
        if 'KMeans_Cluster' not in df_clean.columns and 'DBSCAN_Cluster' not in df_clean.columns:
            # Aligner les indices si pas déjà fait
            export_df = export_df.loc[df_clean.index]
        export_df['IsAnomaly'] = df_clean['IsAnomaly']
        export_df['AnomalyScore'] = df_clean['AnomalyScore']
    
    # Export en CSV
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les résultats (CSV)",
        data=csv,
        file_name="resultats_detection_anomalies.csv",
        mime="text/csv",
    )
    
    # Dashboard synthétique
    st.subheader("📊 Dashboard de synthèse")
    
    # Création d'une vue synthétique
    with st.container():
        st.markdown("### Vue d'ensemble des résultats")
        
        # Création d'une disposition en grille
        col1, col2 = st.columns(2)
        
        with col1:
            # Résumé des algorithmes et leurs résultats
            summary_data = []
            
            if 'KMeans_Cluster' in df_clean.columns:
                n_kmeans_clusters = len(df_clean['KMeans_Cluster'].unique())
                summary_data.append({
                    "Algorithme": "K-Means",
                    "Clusters détectés": n_kmeans_clusters,
                    "Points classifiés": df_clean.shape[0]
                })
            
            if 'DBSCAN_Cluster' in df_clean.columns:
                n_dbscan_clusters = len(set(df_clean['DBSCAN_Cluster'])) - (1 if -1 in df_clean['DBSCAN_Cluster'].values else 0)
                n_noise = list(df_clean['DBSCAN_Cluster']).count(-1)
                summary_data.append({
                    "Algorithme": "DBSCAN",
                    "Clusters détectés": n_dbscan_clusters,
                    "Points classifiés": df_clean.shape[0] - n_noise,
                    "Points de bruit": n_noise
                })
            
            if 'IsAnomaly' in df_clean.columns:
                n_anomalies = df_clean['IsAnomaly'].sum()
                summary_data.append({
                    "Algorithme": "Isolation Forest",
                    "Anomalies détectées": n_anomalies,
                    "Points normaux": df_clean.shape[0] - n_anomalies,
                    "Taux d'anomalies": f"{(n_anomalies / df_clean.shape[0] * 100):.2f}%"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.write(summary_df)
        
        with col2:
            # Graphique récapitulatif des résultats
            if 'IsAnomaly' in df_clean.columns:
                anomaly_counts = df_clean['IsAnomaly'].value_counts().reset_index()
                anomaly_counts.columns = ['Type', 'Count']
                anomaly_counts['Type'] = anomaly_counts['Type'].map({0: 'Normal', 1: 'Anomalie'})
                
                fig = px.pie(anomaly_counts, values='Count', names='Type', 
                            title='Répartition des anomalies',
                            color='Type', color_discrete_map={'Normal': 'blue', 'Anomalie': 'red'})
                st.plotly_chart(fig, use_container_width=True)
            elif 'DBSCAN_Cluster' in df_clean.columns:
                dbscan_counts = df_clean['DBSCAN_Cluster'].value_counts().reset_index()
                dbscan_counts.columns = ['Cluster', 'Count']
                
                fig = px.pie(dbscan_counts, values='Count', names='Cluster', 
                            title='Répartition des clusters DBSCAN')
                st.plotly_chart(fig, use_container_width=True)
    
    # Section d'interprétation des résultats
    st.markdown("### 🔍 Interprétation des résultats")
    
    with st.expander("Guide d'interprétation"):
        st.markdown("""
        #### Comment interpréter les résultats de la détection d'anomalies
        
        1. **K-Means**:
           - Les clusters représentent des groupes de points avec des caractéristiques similaires
           - Les points isolés ou dans de petits clusters peuvent indiquer des comportements atypiques
           - Examinez les caractéristiques des centroïdes pour comprendre ce qui définit chaque cluster
        
        2. **DBSCAN**:
           - Les points de bruit (label -1) sont considérés comme des anomalies
           - L'algorithme identifie des clusters basés sur la densité des points
           - Particulièrement utile pour détecter des attaques distribuées ou des comportements anormaux dans le réseau
        
        3. **Isolation Forest**:
           - Les points identifiés comme anomalies ont un score d'anomalie élevé
           - Efficace pour détecter des observations qui se distinguent facilement du reste des données
           - Examinez les caractéristiques des points avec les scores d'anomalie les plus élevés
        
        #### Types d'anomalies réseau courantes:
        
        1. **Scan de ports**: Tentatives d'identification des services actifs sur un réseau
        2. **Attaques par déni de service (DoS/DDoS)**: Tentatives de surcharge du réseau
        3. **Exfiltration de données**: Transferts anormaux de grandes quantités de données
        4. **Mouvements latéraux**: Tentatives de se déplacer à travers le réseau après une compromission
        5. **Comportements d'utilisateurs anormaux**: Connexions à des heures inhabituelles ou à partir d'emplacements non reconnus
        """)
    
    # Recommandations de sécurité
    st.markdown("### 🛡️ Recommandations de sécurité")
    
    with st.expander("Actions recommandées"):
        st.markdown("""
        #### Actions recommandées suite à la détection d'anomalies
        
        1. **Investigation approfondie**:
           - Analysez les paquets réseau liés aux anomalies détectées
           - Vérifiez les journaux système et applicatifs pour les activités suspectes
           - Corrélation avec d'autres événements de sécurité
        
        2. **Isolation et remédiation**:
           - Isolez temporairement les systèmes suspectés d'être compromis
           - Appliquez les correctifs de sécurité manquants
           - Renforcez les configurations de sécurité
        
        3. **Amélioration du modèle**:
           - Utilisez les feedbacks des analyses pour améliorer le modèle
           - Ajustez les paramètres des algorithmes en fonction des résultats
           - Envisagez d'incorporer des techniques d'apprentissage supervisé si des labels sont disponibles
        
        4. **Mise en place de contre-mesures**:
           - Configurez des règles de détection dans les outils de sécurité existants
           - Implémentez des politiques de sécurité plus strictes pour les comportements identifiés comme risqués
           - Envisagez des solutions automatisées de réponse aux incidents
        """)
    
    # Intégration et déploiement en production
    st.markdown("### 🚀 Intégration et déploiement")
    
    with st.expander("Options de déploiement"):
        deployment_option = st.selectbox(
            "Sélectionnez une option de déploiement:",
            ["Application Streamlit", "API avec FastAPI", "Service Cloud (AWS/GCP/Azure)", "Intégration SIEM"]
        )
        
        if deployment_option == "Application Streamlit":
            st.markdown("""
            #### Déploiement Streamlit
            
            ```bash
            # Installation des dépendances
            pip install -r requirements.txt
            
            # Lancement de l'application
            streamlit run app.py
            ```
            
            Pour déployer sur Streamlit Cloud:
            1. Créez un dépôt GitHub avec votre code
            2. Connectez-vous à [Streamlit Cloud](https://streamlit.io/cloud)
            3. Déployez l'application depuis votre dépôt GitHub
            """)
        
        elif deployment_option == "API avec FastAPI":
            st.markdown("""
            #### Déploiement avec FastAPI
            
            ```python
            # app.py
            from fastapi import FastAPI, UploadFile, File
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import DBSCAN
            import joblib
            
            app = FastAPI()
            
            # Chargez votre modèle pré-entraîné
            model = joblib.load('model/dbscan_model.pkl')
            scaler = joblib.load('model/scaler.pkl')
            
            @app.post("/predict/")
            async def predict(file: UploadFile = File(...)):
                # Traitement du fichier
                df = pd.read_csv(file.file)
                
                # Prétraitement
                df_numeric = df.select_dtypes(include=['number'])
                X_scaled = scaler.transform(df_numeric)
                
                # Prédiction
                predictions = model.fit_predict(X_scaled)
                
                # Formatage des résultats
                results = {
                    "anomalies": int((predictions == -1).sum()),
                    "normal": int((predictions != -1).sum()),
                    "clusters": int(len(set(predictions)) - (1 if -1 in predictions else 0)),
                    "predictions": predictions.tolist()
                }
                
                return results
            
            # Lancement: uvicorn app:app --reload
            ```
            """)
        
        elif deployment_option == "Service Cloud (AWS/GCP/Azure)":
            st.markdown("""
            #### Déploiement sur AWS
            
            1. **AWS Lambda avec API Gateway**:
               - Packagez votre modèle et code
               - Créez une fonction Lambda
               - Configurez API Gateway comme point d'entrée
            
            2. **AWS SageMaker**:
               - Utiliser SageMaker pour entraîner et déployer le modèle
               - Créer un endpoint accessible via API
            
            3. **AWS ECS/EKS**:
               - Conteneurisez votre application avec Docker
               - Déployez sur ECS ou EKS pour une solution scalable
            
            #### Exemple de Dockerfile
            ```dockerfile
            FROM python:3.9-slim
            
            WORKDIR /app
            
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt
            
            COPY . .
            
            EXPOSE 8000
            
            CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
            ```
            """)
        
        elif deployment_option == "Intégration SIEM":
            st.markdown("""
            #### Intégration avec des solutions SIEM
            
            1. **Splunk**:
               - Créez une application Splunk personnalisée
               - Utilisez le SDK Splunk pour Python pour envoyer les alertes
               - Configurez des dashboards pour visualiser les anomalies
            
            2. **Elastic Stack (ELK)**:
               - Envoyez les résultats à Elasticsearch
               - Créez des visualisations Kibana
               - Configurez des alertes basées sur les détections
            
            3. **QRadar/ArcSight**:
               - Utilisez les API REST pour envoyer des événements
               - Configurez des règles de corrélation basées sur les résultats du modèle
            
            ```python
            # Exemple d'intégration avec Elasticsearch
            from elasticsearch import Elasticsearch
            
            es = Elasticsearch(['http://localhost:9200'])
            
            def send_anomaly_to_elk(anomaly_data):
                doc = {
                    'timestamp': datetime.now(),
                    'source_ip': anomaly_data['source_ip'],
                    'destination_ip': anomaly_data['destination_ip'],
                    'score': anomaly_data['anomaly_score'],
                    'algorithm': 'isolation_forest',
                    'features': anomaly_data['features']
                }
                
                res = es.index(index="network-anomalies", document=doc)
                return res
            ```
            """)
    
    # Futur et améliorations
    st.markdown("### 🔮 Améliorations futures")
    
    future_improvements = st.multiselect(
        "Sélectionnez les axes d'amélioration intéressants pour votre projet:",
        [
            "Intégration de données temps réel",
            "Apprentissage incrémental/adaptatif",
            "Ajout d'algorithmes supplémentaires",
            "Optimisation des hyperparamètres automatique",
            "Visualisations avancées",
            "Analyse explicative des anomalies",
            "Déploiement en production",
            "Interface utilisateur plus intuitive"
        ],
        default=["Apprentissage incrémental/adaptatif", "Analyse explicative des anomalies"]
    )
    
    if future_improvements:
        st.write("Axes d'amélioration sélectionnés:")
        
        if "Intégration de données temps réel" in future_improvements:
            st.markdown("""
            - **Intégration de données temps réel**: Utiliser Kafka ou RabbitMQ pour traiter des flux continus de données réseau et détecter les anomalies en temps réel.
            """)
        
        if "Apprentissage incrémental/adaptatif" in future_improvements:
            st.markdown("""
            - **Apprentissage incrémental/adaptatif**: Implémenter des modèles qui s'adaptent au fil du temps aux changements dans les patterns réseau légitimes, réduisant ainsi les faux positifs.
            """)
        
        if "Ajout d'algorithmes supplémentaires" in future_improvements:
            st.markdown("""
            - **Ajout d'algorithmes supplémentaires**: Explorer d'autres algorithmes comme One-Class SVM, Local Outlier Factor, ou des approches basées sur des autoencodeurs.
            """)
        
        if "Optimisation des hyperparamètres automatique" in future_improvements:
            st.markdown("""
            - **Optimisation des hyperparamètres automatique**: Utiliser des techniques comme Grid Search, Random Search ou Bayesian Optimization pour trouver les meilleurs paramètres.
            """)
        
        if "Visualisations avancées" in future_improvements:
            st.markdown("""
            - **Visualisations avancées**: Implémenter des graphes de réseau interactifs, des cartes de chaleur temporelles et d'autres visualisations spécifiques au domaine de la cybersécurité.
            """)
        
        if "Analyse explicative des anomalies" in future_improvements:
            st.markdown("""
            - **Analyse explicative des anomalies**: Développer des techniques pour expliquer pourquoi certains points sont considérés comme des anomalies, facilitant le travail des analystes de sécurité.
            """)
        
        if "Déploiement en production" in future_improvements:
            st.markdown("""
            - **Déploiement en production**: Mettre en place un pipeline CI/CD, gérer les versions du modèle, et implémenter une surveillance des performances du modèle en production.
            """)
        
        if "Interface utilisateur plus intuitive" in future_improvements:
            st.markdown("""
            - **Interface utilisateur plus intuitive**: Améliorer l'UX/UI avec des tableaux de bord personnalisables, des indicateurs visuels simplifiés pour les non-experts et des workflows guidés.
            """)

else:
    # Page d'accueil lorsqu'aucun fichier n'est chargé
    st.markdown("""
    # 🌐 Détection d'Anomalies Réseau par Apprentissage Non Supervisé
    
    ## 🎯 Objectif du projet
    
    Cette application vous permet de détecter des comportements anormaux dans votre trafic réseau en utilisant des algorithmes d'apprentissage non supervisé, notamment:
    
    - **K-Means**: Regroupement basé sur la similarité des caractéristiques
    - **DBSCAN**: Identification de clusters basés sur la densité
    - **Isolation Forest**: Détection d'anomalies basée sur l'isolation
    
    ## 📊 Fonctionnalités
    
    - Prétraitement automatique des données réseau
    - Visualisation des clusters et anomalies
    - Analyse comparative des algorithmes
    - Exportation des résultats
    - Recommandations de sécurité basées sur les anomalies détectées
    
    ## 🚀 Pour commencer
    
    1. Chargez votre fichier CSV contenant les données réseau
    2. Configurez les paramètres des algorithmes
    3. Analysez les résultats et visualisations
    4. Exportez les résultats pour une analyse plus approfondie
    
    ## 📋 Format de données recommandé
    
    Le format idéal est un fichier CSV contenant des caractéristiques de trafic réseau comme:
    - Adresses IP source/destination
    - Ports source/destination
    - Protocole
    - Nombre de paquets/octets
    - Durée des connexions
    - Flags de connexion TCP
    
    Des jeux de données publics comme NSL-KDD, CICIDS2017 ou UNSW-NB15 sont également compatibles.
    
    ## 🔒 Confidentialité
    
    Toutes les données sont traitées localement et ne sont pas partagées à des tiers.
    """)
    
    # Exemple de dataset
    st.markdown("### ⬇️ Télécharger un exemple de dataset")
    
    sample_data = """
    src_ip,dst_ip,src_port,dst_port,protocol,packets,bytes,duration,flags
    192.168.1.1,10.0.0.1,53210,80,TCP,12,1020,0.34,SYN-ACK
    192.168.1.2,10.0.0.2,45123,443,TCP,8,860,0.25,SYN-ACK
    192.168.1.1,10.0.0.3,53211,53,UDP,1,76,0.02,
    192.168.1.3,10.0.0.4,22,22,TCP,350,25420,124.5,ACK
    192.168.1.4,10.0.0.5,33,1433,TCP,15420,1245210,3600.5,ACK
    192.168.1.5,10.0.0.6,44424,80,TCP,6,780,0.15,SYN-ACK
    192.168.1.6,10.0.0.7,51231,443,TCP,9,950,0.28,SYN-ACK
    192.168.1.7,10.0.0.8,32451,53,UDP,1,82,0.03,
    """
    
    st.download_button(
        label="📥 Télécharger un exemple de dataset",
        data=sample_data,
        file_name="sample_network_data.csv",
        mime="text/csv",
    )