#必要なライブラリをインポート
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#学習データとテストデータを同条件でクレンジングする。
#Climate_Data_Preprocessing
def CDP(c_df,c_t_df):
    
    #正規表現を用いて、タブ等の不要な文字列をピンポイントで除去する。
    for df in [c_df,c_t_df]:
        for column in ['昼間天気','夜間天気']:
            for index in list(df.index[df[column].str.contains('\u3000')==True]):
                df[column][index] = re.sub('\s+','',df[column][index])
        
    #ラベルエンコーダーに天気の数値化を学習させる。
    #注1：テストデータも同様の変換をしないとテストの意味がなくなってしまう為、必ず学習データと合わせてfitさせる。
    #注2:LE_AM,LE_PMは天気予測AIでデコーダーとして使うので、グローバル変数として再利用できるようにしておく。
    global LE_AM,LE_PM
    LE_AM,LE_PM = LabelEncoder(),LabelEncoder()
    LE_AM.fit(pd.concat([c_df['昼間天気'],c_t_df['昼間天気']]))
    LE_PM.fit(pd.concat([c_df['夜間天気'],c_t_df['夜間天気']]))
    
    #ラベルエンコーディングで天気を数値データに変換する。
    for A,P in zip(['昼間天気'],['夜間天気']):
        c_df[A] = LE_AM.transform(c_df[A])
        c_df[P] = LE_PM.transform(c_df[P])
        c_t_df[A] = LE_AM.transform(c_t_df[A])
        c_t_df[P] = LE_PM.transform(c_t_df[P])
    
    #置換で余計な文字を取り除く。
    for name in list(c_df.columns)[1:-2]:
        c_df[name] = c_df[name].str.translate(str.maketrans({'-':'',')':'',' ':'','Ã':'',']':'','×':''}))
        c_df[name] = c_df[name].replace('',0)
        c_t_df[name] = c_t_df[name].str.translate(str.maketrans({'-':'',')':'',' ':'','Ã':'',']':'','×':''}))
        c_t_df[name] = c_t_df[name].replace('',0)
    
    #データ型を統一する。
    c_df = c_df.astype('float64')
    c_t_df = c_t_df.astype('float64')
    
    return c_df,c_t_df
 

#正常に動作するかテストする。
#前回のCS関数で予めスクレイピングしておいたデータをデータフレームに変換する。
c_df = pd.DataFrame(c_list)
c_t_df = pd.DataFrame(c_t_list)

#CDP(学習データ,テストデータ)
c_df,c_t_df = CDP(c_df,c_t_df)
display(c_df)
display(c_t_df)

#ラベルエンコーダーも正常に学習されているかテストする。
display(LE_AM.classes_)
