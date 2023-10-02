import sklearn as sk
import pandas as pd
import numpy as np
import gzip
import math
from collections import Counter
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import euclidean_distances
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


#nltk.download('punkt')

df = pd.read_csv("classic4.csv")
#df = pd.read_csv("classic4.csv")

#default_wt = nltk.word_tokenize
#print(df.loc[0]['Text'])

#w1_token = default_wt(df.loc[0]['Text'])
#print(df.head())
X_total = df['text'].tolist()
Y_total = df['class'].tolist()



def pre_process(text):
    tokens = word_tokenize(text.lower())  #tokeniza e bota em minuscula
    new_tokens = [word for word in tokens if word not in stop_words]  #Remove stopwords
    stem_token = [stemmer.stem(word) for word in new_tokens]  #faz o stemming
    return " ".join(stem_token)   #devolve a string pre-processada inteira


def dist_euclid(v1, v2):
    return np.linalg.norm(v1 - v2)

class KNN(BaseEstimator, ClassifierMixin):
    def __init__(self,N=5):
        self.N = N

    def fit(self, x, y):
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.03, preprocessor=pre_process)
        self.X_train = self.vectorizer.fit_transform(x).toarray()
        self.y_train = y
        self.classes_ = unique_labels(y)                

        return self

    def predict(self, x):
        y_pred = []
        X_test_vec = self.vectorizer.transform(x).toarray()

        for test_x in X_test_vec:
            #distancia euclidiana de todos os elementos em relação

            #print(self.X_train[0])
            #distancias = euclidean_distances(test_x, self.X_train)
            distancias = [dist_euclid(test_x, train_x) for train_x in self.X_train]
            #sort pra pegar os N primeiros e devolve os indices desse n primeiros
            k_x_ind = np.argsort(distancias)[:self.N]
            #pega as N primeira labels
            k_labels = [self.y_train[i] for i in k_x_ind]
            # Pega  a label mais recorrente usando o Counter        
            common = Counter(k_labels).most_common(1)
            y_pred.append(common[0][0])
        return y_pred
    
print("")

class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_probs = {}  # P(class)
        self.word_probs = defaultdict(lambda: defaultdict(float))  # P(word | class)

    def fit(self, X, Y):

        self.classes_ = unique_labels(Y)                

        # Count the number of documents in each class
        class_counts = defaultdict(int)
        for label in Y:
            class_counts[label] += 1

        # Calculate P(class)
        total_documents = len(Y)
        for label, count in class_counts.items():
            self.class_probs[label] = count / total_documents

        # Initialize word counts for each class
        word_counts = defaultdict(lambda: defaultdict(int))

        # Count words in each document for each class
        for i, label in enumerate(Y): 
            doc_features = X[i].toarray()[0]      #lista com word count
            for j, count in enumerate(doc_features):  #j é a palavra
                word_counts[label][j] += count
        
        #print(word_counts)

        # Calculate P(word | class)
        #class_word_counts = defaultdict(int)

        for label, counter_palavra_individual in word_counts.items():
            total_words = sum(counter_palavra_individual.values())
            for word, count in counter_palavra_individual.items():
                self.word_probs[label][word] = count / total_words
               # class_word_counts[label] += count
            
    def predict(self, X):
            predictions =[]
            #print(self.word_probs['cacm'])
            #print(self.class_probs)
            for documento in X:
                class_scores = {}        
                # Calculate class scores
                #print(documento.toarray()[0])
                for label, class_prob in self.class_probs.items():
                    score = math.log(class_prob*100)
                    #print(score)
                    #doc_features = X[0].toarray()[0]      #lista com word count
                    #print(doc_features)
                    for i, word in enumerate(documento.toarray()[0]):
                            #print(word)
                            score += math.log(self.word_probs[label][i]*100 + 1) *word
                    
                 
                    class_scores[label] = score
                #print(class_scores)
                predicted_class = max(class_scores, key=class_scores.get)
                predictions.append(predicted_class)
            return predictions
                    

class KNN_Gzip(BaseEstimator, ClassifierMixin):
    def __init__(self,N=5):
        self.N = N
    def fit(self, xtrain, ytrain):
        self.x_train = xtrain
        self.y_train = ytrain
        self.classes_ = unique_labels(y_train)                

    def predict(self,x_test):
        y_train = np.array(self.y_train)
        predict_list =[]
        for x1 in x_test:
            Cx1 = len( gzip.compress(x1.encode()))
            distance_from_x1 = []
            for x2 in self.x_train:
                Cx2 = len( gzip.compress(x2.encode()))
                x1x2 = " ". join([x1 , x2 ])
                Cx1x2 = len( gzip.compress( x1x2.encode() ))
                ncd = ( Cx1x2 - min ( Cx1 , Cx2 ) ) / max (Cx1 , Cx2 )
                distance_from_x1.append( ncd )
            sorted_idx = np.argsort ( np.array(distance_from_x1 ) )
            top_k_class = list(y_train[ sorted_idx[:self.N]])
            predict_class = max(set(top_k_class),key = top_k_class.count)
            predict_list.append(predict_class)
        return predict_list

    





stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

#X_total = [pre_process(x_text) for x_text in X_total]    #Pre-processa X



#X_total = vectorizer.fit_transform(X_total)

#print(vectorizer.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(X_total,Y_total, test_size=0.33, random_state=42)


GK = KNN_Gzip(5)
GK.fit(X_train,y_train)
y_pred = GK.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("KGZIP acuracy")
print(accuracy)

score = cross_val_score(GK, X_total,Y_total,cv=4, scoring='f1_macro')
print("Knn-gzip")
print(score)



knn_teste = KNN(7)
knn_teste.fit(X_train,y_train)
y_pred = GK.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNN acuracy")
print(accuracy)

score = cross_val_score(knn_teste, X_total,Y_total,cv=4, scoring='f1_macro')
print("knn score")
print(score)

vectorizer = CountVectorizer(max_df=0.95, min_df=0.03,preprocessor=pre_process)

NB = NaiveBayes()
vectorizer.fit(X_total)
NB.fit(vectorizer.transform(X_train),y_train)
y_pred = NB.predict(vectorizer.transform(X_test))
accuracy = accuracy_score(y_test, y_pred)
print("NB acuracy")
print(accuracy)

score = cross_val_score(NB, vectorizer.transform(X_total),Y_total,cv=4, scoring='f1_macro')
print("Naive score")
print(score)
#

