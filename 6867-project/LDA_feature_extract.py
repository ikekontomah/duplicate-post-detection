from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import query_cqadupstack as qcqa
import numpy as np
import scipy
import random
import sys
from dataset import DatasetContainer
# Models
from svm import SVM
from random_forest import RandomForest
from pytorch_nn import NeuralNetwork
from k_nearest_neighbors import KNearestNeighbors

from random import shuffle
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import pickle

def extract_data(subforum):
    """
    Produces training and validation data collections for a particular subforum.
    
    Each dataset consists of a list of IDs, a list of duplicate posts, and a term-document matrix corresponding 
    to the post bodies in the dataset.

    Param:
        - subforum: a string containing the subforum name

    Return:
        - train_collection, a data structure containing:
            - train_IDs: list of document IDs contained within the train data
            - train_dups: list duplicate post entries, where each entry is [post1, post2, dup]
            - train_tf: term-document matrix 
        - valid_collection, a data structure containing:
            - valid_IDs: document IDs
            - valid_dups: duplicate post (pairs?)
            - valid_tf: term-document matrix 

    """

    print("Generating Data...")
    print("Subforum: %s" % subforum)


    # populate train and valid collections
    train_collection, valid_collection = construct_train_valid(subforum)

    # initialize tf matrices
    t_length = len(train_collection.bodies)
    tf, feature_names = vectorize(np.concatenate((train_collection.bodies, valid_collection.bodies)))
    train_tf_body, valid_tf_body = np.split(tf.toarray(), [t_length]) 
    
    t_length = len(train_collection.titles)
    tf, feature_names = vectorize(np.concatenate((train_collection.titles, valid_collection.titles)))
    train_tf_title, valid_tf_title = np.split(tf.toarray(), [t_length]) 
    
    t_length = len(train_collection.combined)
    tf, feature_names = vectorize(np.concatenate((train_collection.combined, valid_collection.combined)))
    train_tf_both, valid_tf_both = np.split(tf.toarray(), [t_length]) 

    train_collection.tf_body = train_tf_body 
    train_collection.tf_title = train_tf_title
    train_collection.tf_combined = train_tf_both
    valid_collection.tf_body = valid_tf_body 
    valid_collection.tf_title = valid_tf_title
    valid_collection.tf_combined = valid_tf_both 

    return train_collection, valid_collection


def extract_test_data(subforum, testflag="small"):
    """
    Converts test data from a particular subforum into a collections container.

    """

    print("Starting extraction of test data...")

    # populate train and valid collections
    data_directory = "../data/cqadupstack/" + subforum + ".zip"
    o = qcqa.load_subforum(data_directory)

    unique_test_ID_set = set()

    test_dup = []
    with open(subforum+"_testpairs_"+testflag+".txt") as f:
        for line in f:
            data = line.split(" ")
            (post1, post2, dup) = (unicode(data[0],"utf-8"), unicode(data[1],"utf-8"), unicode(data[2],"utf-8"))
            unique_test_ID_set.add(post1)
            unique_test_ID_set.add(post2)
            test_dup += [[post1,post2,dup]]
            
    unique_test_ID_list = list(unique_test_ID_set)

    # Makes list of post bodies for test data
    test_post_body_list = [o.perform_cleaning(o.get_postbody(ID),remove_stopwords=True) for ID in unique_test_ID_list]
    test_post_title_list = [o.perform_cleaning(o.get_posttitle(ID),remove_stopwords=True) for ID in unique_test_ID_list]
    test_post_both_list = [o.perform_cleaning(o.get_post_title_and_body(ID),remove_stopwords=True) for ID in unique_test_ID_list]
    test_reputation = {ID:(o.get_user_reputation(o.get_postuserid(ID)) if o.get_postuserid(ID)!=False else 0) for ID in unique_test_ID_list}
    test_scores = {ID:o.get_postscore(ID) for ID in unique_test_ID_list}

    test_collection = DatasetContainer(unique_test_ID_list, test_dup, test_post_body_list, test_post_title_list, test_post_both_list, None, None, None,test_reputation,test_scores)
   
    # fill in tf matrices 
    test_tf_body, feature_names = vectorize(test_collection.bodies)
    test_tf_title, feature_names = vectorize(test_collection.titles)
    test_tf_both, feature_names = vectorize(test_collection.combined)

    test_collection.tf_body = test_tf_body 
    test_collection.tf_title = test_tf_title
    test_collection.tf_combined = test_tf_both

    return test_collection


def featurize(train_collection, valid_collection, lda_type="body"):
    """
    Updates the data collections for train and validation with the final train and validation
    matrix data, which can be passed into a classifier.

    Currently, it uses LDA + cosine similarity to convert the post data into vector format, and ultimately 
    into the matrix data.

    Param:
        - train_collection, a data structure containing:
            - train_IDs: list of document IDs contained within the train data
            - train_dups: list duplicate post entries, where each entry is [post1, post2, dup]
            - train_tf: term-document matrix 
        - valid_collection, a data structure containing:
            - valid_IDs: document IDs
            - valid_dups: duplicate post (pairs?)
            - valid_tf: term-document matrix 
        - lda_type: type of LDA modeling we want to do. Must be one of {body, all}

    Return:
        - lda_body: trained LDA model for body
        - lda_title: ditto for title
        - lda_combined: ditto for combined

    """
    train_doc2vec = True
    if train_doc2vec == True:
        print("Training Doc2Vec...")
        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
        doc_training = []
        doc_training += [LabeledSentence(train_collection.bodies[i].split(' '), ["id"+str(train_collection.ids[i])]) for i in range(len(train_collection.ids))]
        doc_training += [LabeledSentence(valid_collection.bodies[i].split(' '), ["id"+str(valid_collection.ids[i])]) for i in range(len(valid_collection.ids))]

        model.build_vocab(doc_training)
        for epoch in range(10):
            print("Epoch %s" % epoch)
            shuffle(doc_training)
            model.train(doc_training, total_examples=model.corpus_count, epochs=model.iter)
        model.save("trained_doc2vec.d2v")
    else:
        print("Loading Doc2Vec...")
        model = Doc2Vec.load('trained_doc2vec.d2v')

    d2v_train = {train_collection.ids[i]:model.docvecs["id"+str(train_collection.ids[i])] for i in range(len(train_collection.ids))}
    d2v_valid = {valid_collection.ids[i]:model.docvecs["id"+str(valid_collection.ids[i])] for i in range(len(valid_collection.ids))}

    # Get cosine distance for doc2vecs
    print("Calculating Cosine Distance Between Doc2Vec Features...")
    train_features = []
    valid_features = []
    d2d_train_cosine, Y_train = cosine_dist(train_collection.dups, d2v_train)
    d2d_valid_cosine, Y_val = cosine_dist(valid_collection.dups, d2v_valid)
    train_features += [d2d_train_cosine]
    valid_features += [d2d_valid_cosine]

    print("Making Topic Model...")

    lda_body = None
    lda_title = None
    lda_combined = None

    # Run LDA for train post bodies
    (lda_body, lda_body_train_cosine, lda_body_valid_cosine) = lda_cosine_similarity(train_collection.tf_body,valid_collection.tf_body, train_collection,valid_collection)

    train_features += [lda_body_train_cosine]
    valid_features += [lda_body_valid_cosine]

    if lda_type=="all":
        # NEW! Run LDA for train post titles
        (lda_title, lda_title_train_cosine, lda_title_valid_cosine) = lda_cosine_similarity(train_collection.tf_title,valid_collection.tf_title,train_collection,valid_collection)

        train_features += [lda_title_train_cosine]
        valid_features += [lda_title_valid_cosine]
        
        # NEW! Run LDA for train post titles and bodies
        (lda_combined, lda_combined_train_cosine, lda_combined_valid_cosine) = lda_cosine_similarity(train_collection.tf_combined,valid_collection.tf_combined,train_collection,valid_collection)

        train_features += [lda_combined_train_cosine]
        valid_features += [lda_combined_valid_cosine]

    #user_reputation = True
    #if user_reputation==True:
    train_reputation = [[train_collection.reputations[dup[0]],train_collection.reputations[dup[1]]] for dup in train_collection.dups]
    valid_reputation = [[valid_collection.reputations[dup[0]],valid_collection.reputations[dup[1]]] for dup in valid_collection.dups]
    train_features += [np.array(train_reputation)]
    valid_features += [np.array(valid_reputation)]

    #post_score = True
    #if post_score ==True:
    train_score = [[train_collection.scores[dup[0]],train_collection.scores[dup[1]]] for dup in train_collection.dups]
    valid_score = [[valid_collection.scores[dup[0]],valid_collection.scores[dup[1]]] for dup in valid_collection.dups]
    train_features += [np.array(train_score)]
    valid_features += [np.array(valid_score)]

    X_train = np.concatenate(train_features, axis=1)
    X_val = np.concatenate(valid_features, axis=1)
    print(X_train.shape)
    print(X_val.shape)

    train_collection.X = X_train
    train_collection.Y = Y_train
    valid_collection.X = X_val
    valid_collection.Y = Y_val

    return (lda_body, lda_title, lda_combined)


def featurize_test(test_collection,models,lda_type="body"):
    """
    Updates the test data collection with the final test 
    matrix data, which can be passed into a classifier.

    Param:
        - train_collection, a data structure containing:
        - models: a group of models, implementation-specific, generated during training
        - lda_type: type of LDA modeling we want to do. Must be one of {body, all}

    Return:
        - nothing

    """

    print("Starting featurization of test data...")

    (lda_body, lda_title, lda_combined) = models

    test_features = []
    
    # doc2vec feature construction
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
    doc_training = []
    doc_training += [LabeledSentence(test_collection.bodies[i].split(' '), ["id"+str(test_collection.ids[i])]) for i in range(len(test_collection.ids))]
    model.build_vocab(doc_training)
    for epoch in range(10):
        print("Epoch %s" % epoch)
        shuffle(doc_training)
        model.train(doc_training, total_examples=model.corpus_count, epochs=model.iter)

    d2v_test = {test_collection.ids[i]:model.docvecs["id"+str(test_collection.ids[i])] for i in range(len(test_collection.ids))}
    d2d_test_features, Y_test = cosine_dist(test_collection.dups, d2v_test)
    test_features += [d2d_test_features]

    # Run LDA for test post bodies
    # NOTE: currently using the same LDA models that were used to test
    lda_body_test_features = lda_cosine_similarity_test(test_collection.tf_body,test_collection,lda_body)
    test_features += [lda_body_test_features]

    if lda_type=="all":
        # NEW! Run LDA for test post titles
        lda_title_test_features = lda_cosine_similarity_test(test_collection.tf_title,test_collection,lda_title)
        test_features += [lda_title_test_features]
        
        # NEW! Run LDA for test post titles and bodies
        lda_combined_test_features = lda_cosine_similarity_test(test_collection.tf_combined,test_collection,lda_combined)
        test_features += [lda_combined_test_features]

    test_reputation = [[test_collection.reputations[dup[0]],test_collection.reputations[dup[1]]] for dup in test_collection.dups]
    test_features += [np.array(test_reputation)]

    test_score = [[test_collection.scores[dup[0]],test_collection.scores[dup[1]]] for dup in test_collection.dups]
    test_features += [np.array(test_score)]

    X_test = np.concatenate(test_features, axis=1)

    test_collection.X = X_test
    test_collection.Y = Y_test


def train_model(model_name,X_train,Y_train,X_val,Y_val):
    """
    Trains a supervised classifier using the training data provided, and scores
    it using the validation dataset.

    Param:
        - model_name: a string containing a model type
        - Train data:
            - X_train
            - Y_train
        - Validation data:
            - X_val
            - Y_val

    Return:
        - model: a supervised classifier, to be used for testing
    """

    if model_name=='svm':
        model = SVM()
    elif model_name=="random_forest":
        max_depth=2
        model = RandomForest(max_depth)
    elif model_name=="neural_network":
        max_depth=2
        out_size = 2
        hidden_size = 30
        in_size = X_train.shape[1]
        model = NeuralNetwork(in_size, hidden_size, out_size)
    elif model_name=="knn":
        n_neighbors = 50
        model = KNearestNeighbors(n_neighbors)
    else:
        return "Error: Model not yet implemented..."

    print("Training " + model_name + "...")

    train_score = model.train(X_train,Y_train)
    valid_score = model.score(X_val,Y_val)

    print("Training Accuracy: %s" % train_score)
    print("Validation Accuracy: %s" % valid_score)

    return model

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def get_duplicate_pairs(subforum):
    data_directory = "../data/cqadupstack/" + subforum + ".zip"
    o = qcqa.load_subforum(data_directory)

    print("collecting duplicate pairs...")
    
    # Step 1: get all unique pairs
    duplicate_pairs = set()
    posts_with_duplicates = o.get_posts_with_duplicates()    
    for post in posts_with_duplicates:
        duplicates = o.get_duplicates(post)
        for dup_candidate in duplicates:
            min_post = unicode(str(min(int(post), int(dup_candidate))),"utf-8")
            max_post = unicode(str(max(int(post), int(dup_candidate))),"utf-8")
            duplicate_pairs.add((min_post,max_post))


    num_duplicate_pairs = len(duplicate_pairs)
    print("Total number of unique duplicate pairs is:", len(duplicate_pairs))

    return duplicate_pairs, num_duplicate_pairs


#TODO: Make method take a list of text files instead of subforums
def construct_train_valid(subforum):
    """Constructs the train and valid sets.

    Keyword arguments:
    subforum -- the subforum to construct the datasets if None, uses all subforum
    """
    #TODO: collect all subforums if subforum=None
    data_directory = "../data/cqadupstack/" + subforum + ".zip"
    o = qcqa.load_subforum(data_directory)

    print("starting speedy data generation...")
    
    # Step 1: get all unique pairs
    duplicate_pairs, num_duplicate_pairs = get_duplicate_pairs(subforum)

    # Step 2: randomly sample pairs of posts that are not duplicates. Idea is to 
    # have roughly a 1:1 ratio of dup to non-dup posts
    non_duplicate_pairs = set()
    while len(non_duplicate_pairs) < num_duplicate_pairs:
        (post1, post2, tag) = o.get_random_pair_of_posts()
        if tag == 'nondup':
            min_post = unicode(str(min(int(post1), int(post2))),"utf-8")
            max_post = unicode(str(max(int(post1), int(post2))),"utf-8")
            non_duplicate_pairs.add((min_post,max_post))

    print("Total number of non-duplicate pairs is:", len(non_duplicate_pairs))        

    # Step 3: generate train/validation splits

    # randomly choose 1/10th of the data to be the validation
    num_validation = num_duplicate_pairs/10
    num_train = num_duplicate_pairs - num_validation

    print("validate with: ", num_validation, "train with: ", num_train)

    list_duplicate_pairs = np.array(list(duplicate_pairs))
    list_duplicate_pairs = np.concatenate((list_duplicate_pairs, np.ones(shape=(num_duplicate_pairs,1))),axis=1)
    list_non_duplicate_pairs = np.array(list(non_duplicate_pairs))
    list_non_duplicate_pairs = np.concatenate((list_non_duplicate_pairs, np.zeros(shape=(num_duplicate_pairs,1))),axis=1)

    # Choosing the validation indices (for dup, non-dup)
    # the masks are used to help filter out the indices that were already used for validate data
    dup_val_indices = np.random.choice(num_duplicate_pairs, num_validation,replace=False)
    mask_dup = np.ones(num_duplicate_pairs,dtype=bool)
    mask_dup[dup_val_indices] = 0

    nondup_val_indices = np.random.choice(num_duplicate_pairs, num_validation,replace=False)
    mask_nondup = np.ones(num_duplicate_pairs,dtype=bool)
    mask_nondup[nondup_val_indices] = 0

    val_pairs = np.concatenate((list_duplicate_pairs[dup_val_indices],list_non_duplicate_pairs[nondup_val_indices]))
    train_pairs = np.concatenate((list_duplicate_pairs[mask_dup],list_non_duplicate_pairs[mask_nondup]))

    print("validation set size: ", val_pairs.shape, "train set size: ", train_pairs.shape)
    
    # Step 4: create the appropriate outputs for the function
    unique_train_ID_set = set()
    unique_valid_ID_set = set()
    train_dup = []
    valid_dup = []

    for entry in val_pairs:
        (post1, post2, dup) = entry
        dup = int(float(dup))
        unique_valid_ID_set.add(post1)
        unique_valid_ID_set.add(post2)
        valid_dup += [[post1,post2,dup]]
    for entry in train_pairs:
        (post1, post2, dup) = entry
        dup = int(float(dup))
        unique_train_ID_set.add(post1)
        unique_train_ID_set.add(post2)
        train_dup += [[post1,post2,dup]]
    
    print("almost done with speedy data generation, compiling post bodies...")

    unique_train_ID_list = list(unique_train_ID_set)
    unique_valid_ID_list = list(unique_valid_ID_set)

    # NOTE: these variable names don't reflect what is actually going on, so beware...
    train_post_body_list = [o.perform_cleaning(o.get_postbody(ID),remove_stopwords=True) for ID in unique_train_ID_list]
    valid_post_body_list = [o.perform_cleaning(o.get_postbody(ID),remove_stopwords=True) for ID in unique_valid_ID_list]

    train_post_title_list = [o.perform_cleaning(o.get_posttitle(ID),remove_stopwords=True) for ID in unique_train_ID_list]
    valid_post_title_list = [o.perform_cleaning(o.get_posttitle(ID),remove_stopwords=True) for ID in unique_valid_ID_list]
    
    train_post_both_list = [o.perform_cleaning(o.get_post_title_and_body(ID),remove_stopwords=True) for ID in unique_train_ID_list]
    valid_post_both_list = [o.perform_cleaning(o.get_post_title_and_body(ID),remove_stopwords=True) for ID in unique_valid_ID_list]
 
    train_reputation = {ID:(o.get_user_reputation(o.get_postuserid(ID)) if o.get_postuserid(ID)!=False else 0) for ID in unique_train_ID_list}
    valid_reputation = {ID:(o.get_user_reputation(o.get_postuserid(ID)) if o.get_postuserid(ID)!=False else 0) for ID in unique_valid_ID_list}

    train_scores = {ID:o.get_postscore(ID) for ID in unique_train_ID_list}
    valid_scores = {ID:o.get_postscore(ID) for ID in unique_valid_ID_list}

    train_collection = DatasetContainer(unique_train_ID_list, train_dup, train_post_body_list, train_post_title_list, train_post_both_list, None, None, None, train_reputation, train_scores)
    valid_collection = DatasetContainer(unique_valid_ID_list, valid_dup, valid_post_body_list, valid_post_title_list, valid_post_both_list, None, None, None, valid_reputation, valid_scores)

    return (train_collection, valid_collection)
    
def vectorize(post_body_list):
    """
    Converts a list of post bodies into a term-document matrix, whose rows correspond to 
    the documents, and whose columns represent terms that occur in the document.

    Currently, uses the top (no_features) features (words) in the document as features 
    for the term-document matrix

    Params:
        - post_body_list: a list containing post bodies

        Returns:
        - tf: the term-document matrix
        - tf_features: a list of the features used in the term-document matrix. 

    """

    no_features = 1000

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(post_body_list)
    tf_feature_names = tf_vectorizer.get_feature_names()
    return tf, tf_feature_names

def cosine_dist(dups, vector_dict):
    cosine_dups = np.empty((len(dups),2))
    for i in range(len(dups)):
        id1, id2, dup = dups[i]
        (vector1, vector2) = (vector_dict[id1],vector_dict[id2])
        cosine_similarity = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        cosine_dups[i]=np.array([cosine_similarity, dup])

    return cosine_dups[:,0].reshape(-1,1), cosine_dups[:,1]


def lda_cosine_similarity(train_tf_matrix,valid_tf_matrix,train_collection,valid_collection):
    """
    Returns cosine similarity scores for the training and validation sets, where
    the topical vectors come from training an LDA model on a particular term-document matrix.

    Param:
        - train_tf_matrix: term-document matrix for train set (LDA is trained with this matrix)
        - valid_tf_matrix: term-document matrix for valid set
        - train_collection: container for train data
        - valid_collection: container for valid data

    Return:
        - lda: the actual trained LDA model
        - lda_train_cosine: array containing cosine distances for each duplicate pair (train set)
        - lda_valid_cosine: array containing cosine distances for each duplicate pair (valid set)

    """
    no_topics = 20

    # Run LDA for the given term-document matrix
    max_iter_num = 15
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=max_iter_num, learning_method='online', learning_offset=50.,random_state=0)
    X_train = lda.fit_transform(train_tf_matrix)
    train_dict = {train_collection.ids[i]:X_train[i,:] for i in range(len(train_collection.ids))}
    
    # Get vectors for valid set
    X_valid = lda.transform(valid_tf_matrix)
    valid_dict = {valid_collection.ids[i]:X_valid[i,:] for i in range(len(valid_collection.ids))}

    # Get cosine distance for training set and valid set
    print("Calculating Cosine Distance Between LDA Features...")
    lda_train_cosine, _ = cosine_dist(train_collection.dups, train_dict)
    lda_valid_cosine, _ = cosine_dist(valid_collection.dups, valid_dict)

    return (lda, lda_train_cosine, lda_valid_cosine)

def lda_cosine_similarity_test(test_tf_matrix,test_collection,lda_model):
    X_test = lda_model.transform(test_tf_matrix)
    test_dict = {test_collection.ids[i]:X_test[i,:] for i in range(len(test_collection.ids))}
    lda_test_cosine, _ = cosine_dist(test_collection.dups, test_dict)

    return lda_test_cosine

if __name__=="__main__":

    #list_of_subforums=["android","english","gaming","gis","mathematica","physics","programmers","stats","tex","unix","webmasters","wordpress"]
    list_of_subforums=["android","gis","webmasters"]
    for ele in list_of_subforums:
        pickle_collections = True          # whether to pickle the current collection objects
        use_pickled_collections = True     # whether to use the pickled collection objects (bypassing extract_data)
 
        print("Generating Data...")
        print(ele)

        ######################
        # STEP 1: Data extraction (train-valid vectorization)
        if use_pickled_collections:
            filename = ele + "_collections.pkl"
            try:
                f = open(filename, "r")
                train_collection, valid_collection = pickle.load(f)
                f.close()
            except IOError:
                print "collections pickle file not found. will call extract_data to get collections."
                train_collection, valid_collection = extract_data(ele)
        else:
            train_collection, valid_collection = extract_data(ele)

        if pickle_collections:
            filename = ele + "_collections.pkl"
            with open(filename, "w") as f:
                pickle.dump((train_collection, valid_collection),f)

        ######################
        # STEP 2: Featurization using LDA + cosine similarity

        lda_type = "body"

        (lda_body, lda_title, lda_combined) = featurize(train_collection, valid_collection,lda_type)
            
        ########################
        # STEP 3: Model training

        # Train (and select) the appropriate model!
        model_name = 'random_forest'
        print(train_collection.X[:5])
        print(train_collection.Y[:5])
        print(valid_collection.X[:5])
        print(valid_collection.Y[:5])
        model = train_model(model_name,train_collection.X,train_collection.Y,valid_collection.X,valid_collection.Y)

        ########################
        # (Optional) Step 4: Testing!
       
        ready_to_test = True

        if ready_to_test:
            print("Starting testing!")
            
            # vectorize the data 
            test_collection = extract_test_data(ele,"small")

            # featurize the data

            # NOTE: the lda_type flag here must be the same as the lda_type flag for training!
            featurize_test(test_collection,(lda_body, lda_title, lda_combined),lda_type) 

            assert test_collection.X.shape[1]==train_collection.X.shape[1]

            # run model with the feature data!
            test_score = model.score(test_collection.X,test_collection.Y)

            print("Test Accuracy: %s" % test_score)

