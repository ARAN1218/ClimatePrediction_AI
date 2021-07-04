#必要なライブラリをインポート
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


#気温予測AIの学習プログラム(LIGHTGBM)
#Climate_Temperature_Learning_LIGHTGBM
def CTLL(df):
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
    #XGBOOSTの時と要領は同じ。
    params = {'objective':'regression','metrics':'rmse','silent':1,'random_state':71,
              'max_depth':4,
              'min_child_weight':2,
              'gamma':0.5,
              'colsample_bytree':0.8,
              'colsample_bylevel':0.3,
              'subsample':1.0,
              'alpha':1,
              'eta':0.05, 
              'lambda':1}
    num_round = 1000
    early_stopping_rounds = 50

    #LIGHTGBMモデルに学習させる。
    model = lgb.train(params,ltrain,num_boost_round=num_round,early_stopping_rounds=early_stopping_rounds,
                      valid_names=['train','valid'],valid_sets=[ltrain,lvalid])
    #XGBOOSTと違い、予測時に用いるデータを専用形式に変換せずにそのまま用いる。
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


#CS関数で予めスクレイピングしておいたデータをデータフレームに変換する。
c_df = pd.DataFrame(c_list)
c_t_df = pd.DataFrame(c_t_list)
#CDP関数でデータクレンジングする。
c_df,c_t_df = CDP(c_df,c_t_df)

#CTLL(学習データ)
model_l_temp = CTLL(c_df)


#作成したモデルにテストデータを入力してテストする。
target = c_t_df.平均気温
test = c_t_df.drop(['平均気温'],axis=1)
pred = model_l_temp.predict(test)

#モデルの性能を表示させる。
display(pd.concat([target, pd.DataFrame(pred,columns=['prediction'])],axis=1))
print('MAE:',mean_absolute_error(target,pred))
print('MSE:',mean_squared_error(target,pred))
print('RMSE:',np.sqrt(mean_squared_error(target,pred)))


#XGBOOSTと比較すると、予測精度は大差ないが、学習時間が早いGBDTモデルである。
