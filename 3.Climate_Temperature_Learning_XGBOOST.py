#必要なライブラリをインポート
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


#気温予測AIの学習プログラム(XGBOOST)
#Climate_Temperature_Learning_XGBOOST
def CTLX(df):
    #学習データと教師データに分ける。
    target = df['平均気温']
    train = df.drop(['平均気温'],axis=1)
    
    #クロスバリデーションを行う。
    kf = KFold(n_splits=4,shuffle=True,random_state=71)
    tr_idx,va_idx = list(kf.split(train))[0]
    tr_x,va_x = train.iloc[tr_idx],train.iloc[va_idx]
    tr_y,va_y = target.iloc[tr_idx],target.iloc[va_idx]
    
    #データ型をXGBOOST用に適合させる。
    dtrain = xgb.DMatrix(tr_x,label=tr_y)
    dvalid = xgb.DMatrix(va_x,label=va_y)
    
    #ハイパーパラメータチューニングを行う。
    #今回は10年分(2010~2019)のデータに合わせているが、データが変わる度にチューニングしなければならない。
    params = {'objective':'reg:linear','silent':1,'random_state':71,
              'max_depth':4, #3~9を2刻みで試す。
              'min_child_weight':1, #1~5を1刻みで試す。
              'gamma':0.3, #0.0~0.4を試す。
              'colsample_bytree':1.0, #0.6~1.0を0.1刻みで試す。
              'colsample_bylevel':0.3, #0.6~1.0を0.1刻みで試す。
              'subsample':1.0, #0.6~1.0を0.1刻みで試す。
              'alpha':1, #1e-5,1e-2,0.1,1,100を試す。
              'eta':0.15, #0.1から減少させる。(今回は何故か上げた方が良かった。)
              'lambda':1} #変更なし。
    num_round = 1000 #early_stoppingで調整するので、十分な数に設定しておく。
    early_stopping_rounds = 30 #10/eta が相場らしい。

    #XGBOOSTモデルに学習させる。
    watchlist = [(dtrain,'train'),(dvalid,'eval')]
    model = xgb.train(params,dtrain,num_round,early_stopping_rounds=early_stopping_rounds,evals=watchlist)
    va_pred = model.predict(dvalid)
    
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

#CTLX(学習データ)
model_x_temp = CTLX(c_df)


#作成したモデルにテストデータを入力してテストする。
target = c_t_df.平均気温
test = c_t_df.drop(['平均気温'],axis=1)

dtest = xgb.DMatrix(test)
pred = model_x_temp.predict(dtest)

#モデルの性能を表示させる。
display(pd.concat([target, pd.DataFrame(pred,columns=['prediction'])],axis=1))
print('MAE:',mean_absolute_error(target,pred))
print('MSE:',mean_squared_error(target,pred))
print('RMSE:',np.sqrt(mean_squared_error(target,pred)))
