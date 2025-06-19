import pandas as pd 
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata
from .article_quality_functions import date_check
from tqdm import tqdm
import itertools as itr

def filter_dataframe(df_in, year_min, month_min, year_max, month_max):
    timestamps = df_in.index.get_level_values(0)

    # The last state the texts were in
    df_previous = df_in[(
                         (timestamps.year < year_min) | 
                         (
                            (timestamps.year == year_min) & 
                            (timestamps.month < month_min))
                        )].groupby(level=1).tail(1)
    
    # The states the texts took during the timespan
    df_current = df_in[(
                        ((timestamps.year > year_min) & (timestamps.year < year_max))
                        |
                        ((timestamps.year == year_min) & (timestamps.month >= month_min))
                        |
                        ((timestamps.year == year_max) & (timestamps.month <= month_max))
                        )]

    if len(df_previous) == 0:
        df_out = df_current
    else:
        
        df_out = pd.concat((df_previous, df_current), axis=0)
    
    return df_out

def noms_par_periode_et_noeud(attracteur_vecteur, df_in):
    
    labels = np.where((rankdata(cdist(attracteur_vecteur, df_in.to_numpy()),axis=0,method="ordinal")==1))[0]
    #node_freq, count_freq = np.unique(labels, return_counts=True)
    att = {un : [] for un in np.unique(labels)}

    for name, label in zip(df_in.index.get_level_values(1),labels):
        att[label] += [name]

    for un in np.unique(labels):
        att[un] = np.unique(att[un])

    return att, (list(att.keys()), [len(un) for un in att.values()])

def E_matrix():
    """
        Fonction qui récupère la quantité de catégories par noeud pendant une période données
    """
    pass 

def N_matrix():
    """
        Fonction qui récupère la quantité d'articles par noeuds par périodes
    """

    pass 


def articles_to_qual_vect(adq_dict, ba_dict, mois, an, articles):
    out = []
    #print(mois, an)
    for article in articles:
        
        out.append(int( 
            (
                (date_check(adq_dict[article],mois, an) if article in adq_dict.keys() else False) 
                    or 
                (date_check(ba_dict[article],mois, an) if article in ba_dict.keys() else False)
            )
                    )
                        )
    return out 

def relevant_cluster_identifier(df_style : pd.DataFrame,
                                cluster_matrix : NDArray,
                                good_articles : dict,
                                featured_articles : dict) -> NDArray[np.bool_]:
    """Function that gives the relevance of the clusters we study

    Args : 
        df_style : A dataframe containing the stylistic dimensions of the articles and all their revisions
        cluster_matrix : A 2d array containing the centroïd of the clusters (rows) and the value of each dimensions (columns)
        good_articles : A dictionnary, each key is the name of an articles, each value is the date when it got elected good article
        featured_articles : Idem as good_articles but for the date for the label "featured" 
    
    Return:
        relevant_cluster_mask : a mask that determine wheather a cluster is relevant or not based on the fact that it had, for one semester at least, a higher probability than randomness to be a relevant cluster

    """
    # Define the columns name and the ranges of the windows, here it's semesters
    semester_columns = [str(i)+" Jan.-June" if j[0] == 1 else str(i)+" July-Dec." for i,j in itr.product(range(2005,2020),[(1,6),(7,12)])]
    
    posterior_probability = np.zeros(shape=(len(semester_columns), cluster_matrix.shape[0]))
    # Loop over the windows parameters to extract, semester by semester the article-cluster density and informations
    for e,(i,j) in tqdm(enumerate(itr.product(range(2005,2020),[(1,6),(7,12)]))):
        df_filt = filter_dataframe(df_style,i,j[0],i,j[1])
        
        ## Finds to which cluster each article belong
        wh = np.where((rankdata(cdist(cluster_matrix, df_filt.to_numpy()).T,axis=1,method="ordinal")==1))
        articles = wh[0]
        clusters = wh[1]

        ## Remplis pour chaque article(version d'article), le cluster où il se trouve  : shape (N_clusters, N_articles)
        clusters_articles = np.zeros(shape=(cluster_matrix.shape[0],df_filt.shape[0]))
        clusters_articles[clusters,articles] = 1 
        
        ## Encode si une version/article est de qualité ou non : shape (N_articles, 1)
        vector_quality = np.array(articles_to_qual_vect(featured_articles,
                                                        good_articles, 
                                                        j, i, 
                                                        df_filt.index.get_level_values(1)))

        ## Encode the posterior probability of a cluster, given the quality article
        posterior_probability[e, :] = probabilty_computation(clusters_articles, vector_quality)
    
    
    relevant_cluster_mask = np.where(posterior_probability > 1/cluster_matrix.shape[0], True, False).sum(0).astype("bool")
    
    return relevant_cluster_mask

def probabilty_computation(density_matrix : NDArray,
                           quality_matrix : NDArray) -> NDArray:
    """Give the posterior probability of P(Cluster | Quality)

    This function compute the probability that a quality article is contained in a cluster, based on the number of quality articles in the cluster and the number of article in the cluster

    Args:
        density_matrix : a (N_clusters,M_articles) matrix that encode if an article/version is in the cluster
        quality_matrix : a (M_articles) array that encode if an article/version is of quality or not

    Return:
        posterior_probability : a (N_cluster, 1) array that encode the posterior probability of a cluster knowing that we have a quality article    
    """ 

    divider = lambda arr1, arr2 : np.divide(arr1,
                                            arr2,
                                            out = np.zeros_like(arr1, dtype=float),
                                            where= arr2!=0)
    
    assert density_matrix.shape[1] == quality_matrix.shape[0], f"Density matrix and quality matrix shapes do not match !"

    ## Donne les articles de qualités par cluster : shape (N_clusters , 1)
    clusters_quality = np.matmul(density_matrix, quality_matrix[:,np.newaxis])
    assert clusters_quality.shape == (density_matrix.shape[0],1), f"Unexpected dimension cluster_quality = {clusters_quality.shape}"

    ## Encode P(Q|N) : shape (N_clusters, 1)
    likelihood_quality_given_cluster = divider(clusters_quality,density_matrix.sum(1)[:,np.newaxis])
    assert likelihood_quality_given_cluster.shape == (density_matrix.shape[0],1), f"Unexpected dimension likelihood shape = {likelihood_quality_given_cluster.shape}"

    ## Prior P(N) : shape (N_clusters, 1)
    prior_cluster = divider(density_matrix.sum(1),density_matrix.sum())
    assert prior_cluster.shape == (density_matrix.shape[0],), f"Unexpected dimension prior cluster shape = {prior_cluster.shape}"

    ## Numérateur : shape (N_clusters, 1)
    numerateur = likelihood_quality_given_cluster*prior_cluster[:,np.newaxis]
    assert numerateur.shape == (density_matrix.shape[0],1), f"Unexpected dimension numerator shape = {numerateur.shape}"

    ## Dénominateur : shape 1 
    denominateur = numerateur.sum()
    assert denominateur.shape == (), f"Unexpected dimension denominator shape = {denominateur.shape}"

    ## P(N|Q) : shape (N_clusters, 1)
    posterior_probability = numerateur/denominateur 
    assert posterior_probability.shape == (density_matrix.shape[0],1), f"Unexpected dimension posterior shape = {posterior_probability.shape}"

    return np.squeeze(posterior_probability)


def semester_vectorizer(cluster_matrix : NDArray,
                        df_in : pd.DataFrame,
                        df_topic : pd.DataFrame,
                        relevant_nodes_maks : NDArray) -> NDArray:
    
    """Function that output the number of articles in a given node
    
    Args:
        cluster_matrix : The matrix with the position of the centroids
        df_in : the input dataset
        relevant_nodes_masks : The mask that exclude irrelevant nodes

    Return:
        cluster_text_mapping : The articles belonging to each clusters
    """
    
    ## Get which article belong to which cluster
    centroid_dist = cdist(df_in.to_numpy(), cluster_matrix)
    #print(centroid_dist.shape, df_in.shape, cluster_matrix.shape)
    clusters = np.argmin(centroid_dist,axis=1)
    #print(len(clusters), cluster_matrix.shape)
    #wh = np.where((rankdata(cdist(cluster_matrix, df_in.to_numpy()).T,axis=1,method="ordinal")==1))
    #texts = wh[0]
    #cluster = wh[1]

    ## Dictionnaire qui retourne pour chaque attracteur les textes qui leur ressemblent au moment étudié
    #att={}
    articles_names = df_in.index.get_level_values(1)
    #print(len(articles_names))
    cluster_text_mapping = {cluster : articles_names[clusters==cluster] for cluster in range(cluster_matrix.shape[0])}
    vector_out = np.array([df_topic.loc[np.unique(articles),:].to_numpy().sum(0) if len(articles) >0 else np.zeros((df_topic.shape[1])) for key,articles in cluster_text_mapping.items()])
    #print(vector_out.shape)
    #for text_type in np.unique(clusters):
    #   att[text_type] = [articles_names[text] for text in texts[clusters == text_type]]

    #att = {text_type : text for (text,text_type) in zip(texts, text_types)}
    #node_freq, count_freq = np.unique(labels, return_counts=True)


    #for name, label in zip(df_in.index.get_level_values(1),text_type):
    #    att[label] += [name]

    #for un in np.unique(labels):
    #    att[un] = np.unique(att[un])

    return vector_out, (clusters, [len(un) for key,un in cluster_text_mapping.items()])

def vectorizer(df_in : pd.DataFrame,
               df_topic : pd.DataFrame,
               cluster_matrix : NDArray,
               relevant_nodes_mask : NDArray) -> NDArray:
    """Function to aggregate the data semester per semester

    Args:
        df_in : The input dataframe with .
        df_topic : A dataframe with the topic for each article
        cluster_matrix : the matrix with the centroids of each clusters
        relevant_nodes_mask : a mask to remove the irrelevant clusters from the analysis

    Return:
        out : The (n_samples,m_features) matrix, with n_samples being (#semester X #clusters) and m_features = #topics  
    """

    semester_columns = [str(i)+" Jan.-June" if j[0] == 1 else str(i)+" July-Dec." for i,j in itr.product(range(2005,2020),[(1,6),(7,12)])]

    n_samples = len(semester_columns)*sum(relevant_nodes_mask)
    n_features = df_topic.shape[1]

    multi_index = pd.MultiIndex.from_tuples([(semester, cluster) for semester, cluster in itr.product(semester_columns,range(cluster_matrix.shape[0]))])
    df_variables = pd.DataFrame(index=multi_index, columns=list(df_topic.columns)+["target"])
    #out = np.zeros(shape=(n_samples, n_features))

    for e,(i,j) in tqdm(enumerate(itr.product(range(2005,2020),[(1,6),(7,12)]))):
        df_filt = filter_dataframe(df_in,i,j[0],i,j[1])

        out, (clusters, member_size) = semester_vectorizer(cluster_matrix,
                                                df_filt,
                                                df_topic,
                                                relevant_nodes_mask)
        
        vectors = [np.append(vector, size) for vector, size in zip(out,member_size)]
        semester = str(i)+" Jan.-June" if j[0] == 1 else str(i)+" July-Dec."
        #print(np.array(vectors).shape, len([(semester, cluster) for cluster in range(cluster_matrix.shape[0])]) )
        #print(df_variables.shape)
        df_variables.loc[[(semester, cluster) for cluster in range(cluster_matrix.shape[0])],:] = vectors

    return df_variables
#attractors_dict, tuple_jsp = noms_par_periode_et_noeud(NG.position_matrix[np.where(pb)[0]],sampled_textes)


"""
import itertools as itr 

#from functions.article_quality_functions import articles_to_qual_vect
from functions.category_vectorizer import filter_dataframe, category_vectorizer, articles_to_qual_vect
from tqdm import tqdm 

from scipy.stats import rankdata
from scipy.spatial.distance import cdist


dict_qual_nodes = {}
node_density = {}
qual_proportion = {}
raw_mat = {}
for e,(i,j) in tqdm(enumerate(itr.product(range(2005,2020),[(1,6),(7,12)]))):
    df_filt = filter_dataframe(df_style,i,j[0],i,j[1])
    
    ## ça donne le cluster auquel appartient chaque texte
    wh = np.where((rankdata(cdist(cluster_matrix, df_filt.to_numpy()).T,axis=1,method="ordinal")==1))
    #print(wh)
    texts = wh[0]
    text_types = wh[1]

    ## Remplis pour chaque article(version d'artilce), le cluster où il se trouve  : shape (N_clusters, N_articles)
    clusters_articles = np.zeros(shape=(cluster_matrix.shape[0],df_filt.shape[0]))
    clusters_articles[text_types,texts] = 1 
    
    ## Encode si une version/article est de qualité ou non : shape (N_articles, 1)
    vector_quality = np.array(articles_to_qual_vect(dict_featured_articles,
                                                    dict_good_articles, 
                                                    j, i, 
                                                    df_filt.index.get_level_values(1)))

    



    ## Conserve la matrice binaire (N,M) qui dit les paires articles/cluster
    raw_mat[cols[e]] = art_mat
    
    ## Art_mat devient la probabilité d'un article sachant un cluster
    art_mat /= art_mat.sum(0)

    
    ## Donne la probabilité d'un article de qualité par cluster
    dict_qual_nodes[cols[e]] = np.matmul(art_mat, vector_quality[:,np.newaxis])

    ## Sauve art mat et vector quality
    node_density[cols[e]] = art_mat
    qual_proportion[cols[e]] = vector_quality"""