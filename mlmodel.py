import pickle
import math
def Ratings(comments):
    vectorizer = pickle.load(open('./vector.pkl','rb'))
    vectorizer_gan = pickle.load(open('./vectorizer_gan.pkl','rb'))
    matrix = vectorizer.transform(comments)
    matrix_gan = vectorizer_gan.transform(comments)
    model = pickle.load(open('./model.pkl','rb'))
    model_gan = pickle.load(open("./model_gan.pkl", "rb"))
    list_of_ratings = model.predict(matrix)
    count = {
    '1':0,
    '0':0,
    '-1':0
    }
    # list_of_ratings = [1,-1,1,1,0]
    for i in list_of_ratings:
        count[str(i)]+= 1

    # print(count)
    rating =  (count['-1']*0 + count['0']*1 + count['1']*2)*5/(len(list_of_ratings)*2)
    list_of_ratings_gan = model_gan.predict(matrix_gan).astype(int)
    count = {
    '1':0,
    '0':0,
    '-1':0
    }
    # list_of_ratings = [1,-1,1,1,0]
    for i in list_of_ratings_gan:
        count[str(i)]+= 1

    rating_gan =  (count['-1']*0 + count['0']*1 + count['1']*2)*5/(len(list_of_ratings_gan)*2)
    print(rating_gan)
    return round(rating,2)