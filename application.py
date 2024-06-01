# Importation des bibliothèques
import streamlit as st
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import col, concat, lit, lower, udf
from pyspark.sql.types import StringType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql import SparkSession,  functions as F
# Configuration de la barre de recherche
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide",
    page_icon="🎬"
)

st.markdown(page_bg_img, unsafe_allow_html=True)

# Configurer Spark
conf = SparkConf().setAppName("Recommendations") \
                  .set("spark.executor.memory", "6g") \
                  .set("spark.driver.memory", "6g") \
                  .set("spark.network.timeout", "600s") \
                  .set("spark.executor.heartbeatInterval", "60s")

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Lire le fichier en DataFrame en utilisant le schéma inféré automatiquement
data = spark.read.csv('C:\\Users\\HP\\Downloads\\dataFinale', inferSchema=True, header=True)

with st.form("form"):
    # Indication utilisateur
    st.header('Recherche de film par titre, genres, acteurs, année de sortie')
    # Choix de l'utilisateur
    keywords = st.text_input("Entrer les informations ici")
    # Bouton submit
    submit_1 = st.form_submit_button("Rechercher")
    
    if submit_1:
        # Renommer la colonne '\N' en 'duration'
        data_renamed = data.withColumnRenamed(r'\N', 'duration')
        # Remplacer les valeurs manquantes par des espaces vides dans certaines colonnes
        data_filled = data_renamed.fillna(' ')
        colonnes_a_convertir = ['duration', 'description', 'year', 'genre', 'director', 'writer', 'actors']
        
        # Convertir les colonnes en StringType
        for colonne in colonnes_a_convertir:
            data_filled = data_filled.withColumn(colonne, col(colonne).cast(StringType()))

        # Fonction pour supprimer les espaces dans les chaînes de caractères
        def remove_space(text):
            return text.replace(" ", "") if text else ""

        # UDF pour supprimer les espaces
        remove_space_udf = udf(remove_space, StringType())

        # Appliquer la fonction UDF pour supprimer les espaces dans les chaînes de caractères
        data = data_filled.withColumn('genre', remove_space_udf(col('genre')))
        data = data.withColumn('actors', remove_space_udf(col('actors')))
        data = data.withColumn('writer', remove_space_udf(col('writer')))
        data = data.withColumn('director', remove_space_udf(col('director')))
        
        # Créer une nouvelle colonne 'tags' qui contient la concaténation des colonnes pertinentes
        data = data.withColumn('tags', concat(col('description'), lit(' '), col('actors'), lit(' '), col('director'), lit(' '), col('writer'), lit(' '), col('genre'), lit(' '), col('duration'), lit(' '), col('year')))
        # Transformer la colonne 'tags' en minuscule
        data = data.withColumn('tags', lower(col('tags')))
        
        # Proposition de l'algorithme
        # Tokenizer pour transformer 'tags' en 'words'
        tokenizer = Tokenizer(inputCol="tags", outputCol="words")
        words_data = tokenizer.transform(data)
        
        # StopWordsRemover pour enlever les mots vides
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        filtered_data = remover.transform(words_data)
        
        # Calcul des TF pour convertir des données textuelles en vecteurs de caractéristiques
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
        featurized_data = hashing_tf.transform(filtered_data)
        
        # Calcul des IDF pour réduire le poids des termes fréquents
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idf_model = idf.fit(featurized_data)
        vectorized_data = idf_model.transform(featurized_data)
        
        # Définir la UDF pour le calcul de similarité de Jaccard
        def calculate_jaccard_similarity(v1, v2):
            set1 = set(v1)
            set2 = set(v2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return float(intersection) / float(union) if union != 0 else 0.0

        # Enregistrer la UDF
        jaccard_udf = udf(calculate_jaccard_similarity, FloatType())

        # Fonction pour obtenir les recommandations à partir des mots-clés
        def get_recommendations_from_keywords(keywords, top_n=4):
            # Créer un DataFrame avec les mots-clés comme description
            keywords_df = spark.createDataFrame([(0, keywords)], ["imdb_title_id", "tags"])
            # Transformer les mots-clés en vecteur TF-IDF
            words_data = tokenizer.transform(keywords_df)
            filtered_data = remover.transform(words_data)
            featurized_data = hashing_tf.transform(filtered_data)
            tfidf_keywords = idf_model.transform(featurized_data)
            keywords_vector = tfidf_keywords.select("features").collect()[0][0]
            keywords_vector_list = keywords_vector.toArray().tolist()
            # Ajouter les colonnes de similarité
            similarity_data = vectorized_data.withColumn("similarity", jaccard_udf(F.col("features"), F.lit(keywords_vector_list)))

            # Trier et sélectionner les films les plus similaires
            top_similar_movies = similarity_data.orderBy(F.col("similarity").desc()).select("title", "original_title", "tags").limit(top_n)
            return top_similar_movies

        suggestion = get_recommendations_from_keywords(keywords, top_n=4)
        st.write(suggestion.collect())

# SOUS-TITRE
st.subheader("Bon Visionnage ! 🍿")
