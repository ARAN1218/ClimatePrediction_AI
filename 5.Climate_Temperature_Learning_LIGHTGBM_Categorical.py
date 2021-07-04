#必要なライブラリをインポート
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


#気温予測AIの学習プログラム(LIGHTGBM)(カテゴリ変数をそのまま使うver)
#Climate_Temperature_Learning_LIGHTGBM_Categorical
def CTLLC(df):
    #学習データと教師データに分ける。
    target = df['平均気温']
    train = df.drop(['平均気温'],axis=1)
    
    #クロスバリデーションを行う。
    kf = KFold(n_splits=4,shuffle=True,random_state=71)
    tr_idx,va_idx = list(kf.split(train))[0]
    tr_x,va_x = train.iloc[tr_idx],train.iloc[va_idx]
    tr_y,va_y = target.iloc[tr_idx],target.iloc[va_idx]
    
    #データ型をLIGHTGBM用に適合させる。
    ltrain = lgb.Dataset(tr_x,tr_y)
    lvalid = lgb.Dataset(va_x,va_y)
    
    #ハイパーパラメータチューニングを行う。
    #今回は天気をカテゴリ変数として扱うので、先程と同じチューニングでは最適ではないことに注意する。
    params = {'objective':'regression','metrics':'rmse','silent':1,'random_state':71,
              'max_depth':3,
              'min_child_weight':4,
              'gamma':0.2,
              'colsample_bytree':0.7,
              'colsample_bylevel':0.3,
              'subsample':0.07,
              'alpha':1,
              'eta':0.1, 
              'lambda':1}
    num_round = 1000
    early_stopping_rounds = 50

    #LIGHTGBMモデルに学習させる。
    #categorical_featureに天気を設定し、天気をカテゴリ変数
    categorical_feature = ['昼間天気','夜間天気']
    model = lgb.train(params,ltrain,num_boost_round=num_round,
                      early_stopping_rounds=early_stopping_rounds,
                      valid_names=['train','valid'],valid_sets=[ltrain,lvalid],
                     categorical_feature=categorical_feature)
    va_pred = model.predict(va_x)
    
    #テストデータとその予測結果を表示させる。
    #見やすくする為に、concatで横に実値と予測値を結合させる。
    #実値のindexがめちゃくちゃだったので、reset_indexメソッドでindexをリセットしてから結合させた。
    va_y_df = pd.DataFrame(va_y).reset_index(drop=True)
    va_pred_df = pd.DataFrame(va_pred,columns=['prediction'])
    display(pd.concat([va_y_df,va_pred_df],axis=1))
    
    #モデルの性能を表示させる。
    print('MAE:',mean_absolute_error(va_y,va_pred))
    print('MSE:',mean_squared_error(va_y,va_pred))
    print('RMSE:',np.sqrt(mean_squared_error(va_y,va_pred)))
    
    #学習したモデルを返す。
    return model




#必要なライブラリをインポート
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#学習データとテストデータを同条件で前処理する。(カテゴリ変数をそのまま使うver)
#Climate_Data_Preprocessing_Categorical
def CDPC(c_df,c_t_df):
    
    #天気をカテゴリ変数として扱うので、ラベルエンコーディングはしない。
    
    #置換で余計な文字を取り除く。
    #天気以外のデータのデータ型を統一する。
    for name in list(c_df.columns)[1:-2]:
        c_df[name] = c_df[name].str.translate(str.maketrans({'-':'',')':'',' ':'','Ã':'',']':'','×':''}))
        c_df[name] = c_df[name].replace('',0)
        c_df[name] = c_df[name].astype('float64')
        c_t_df[name] = c_t_df[name].str.translate(str.maketrans({'-':'',')':'',' ':'','Ã':'',']':'','×':''}))
        c_t_df[name] = c_t_df[name].replace('',0)
        c_t_df[name] = c_t_df[name].astype('float64')
    
    #カテゴリ変数だけはfloatではなくcategory型に変換する。
    c_df['昼間天気'] = c_df['昼間天気'].astype('category')
    c_df['夜間天気'] = c_df['夜間天気'].astype('category')
    c_t_df['昼間天気'] = c_t_df['昼間天気'].astype('category')
    c_t_df['夜間天気'] = c_t_df['夜間天気'].astype('category')
    
    #クレンジングしたデータを返す。
    return c_df,c_t_df


#CS関数で予めスクレイピングしておいたデータをデータフレームに変換する。
c_df = pd.DataFrame(c_list)
c_t_df = pd.DataFrame(c_t_list)
#CDP関数ではなく、CDPC関数でデータクレンジングする。
#CDPC(学習データ,テストデータ)
c_df,c_t_df = CDPC(c_df,c_t_df)

#CTLLC(学習データ)
model_l_c_temp = CTLLC(c_df)
#天気をきちんとcategory型にしているのに警告が出る(恐らく予測には影響が無いと思われる)。


#作成したモデルにテストデータを入力してテストする。
target = c_t_df.平均気温
test = c_t_df.drop(['平均気温'],axis=1)
pred = model_l_c_temp.predict(test)

#モデルの性能を表示させる。
display(pd.concat([target, pd.DataFrame(pred,columns=['prediction'])],axis=1))
print('MAE:',mean_absolute_error(target,pred))
print('MSE:',mean_squared_error(target,pred))
print('RMSE:',np.sqrt(mean_squared_error(target,pred)))

#学習データに対する評価値は良かったが、未知のデータに対しての評価値は悪い。
#lightgbmのカテゴリ変数をそのまま学習に使う方法は、過学習しやすくなると考えられる。
