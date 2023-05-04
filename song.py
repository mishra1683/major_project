import pandas as pd

def abc():
    m=pd.read_csv('dataset/file2.csv')
    NUM_RECOMMEND=10
    happy_set=[]
    happy_set.append(m[m['kmeans']==0]['song_name'].sample(n=NUM_RECOMMEND))
    return pd.DataFrame(happy_set).T