#必要なライブラリをインポートする。
from bs4 import BeautifulSoup
import requests
from time import sleep
import pandas as pd


#横浜の気象情報を気象庁から年単位でスクレイピングする。
#Climate_Scraping
def CS(start,end):
    
    #スクレイピングした情報を保存する為のリストを用意しておく。
    c_list = []
    
    #アクセスする気象情報が記載されているページのurlを用意しておく。
    url = 'https://www.data.jma.go.jp/stats/etrn/view/daily_s1.php?prec_no=46&block_no=47670&year={}&month={}&view='
    
    #引数として入力した取得したい年の範囲だけfor文で回す。
    for year in range(start,end+1):
        for month in range(1,13):
            
            #このターンでスクレイピングするurlを作成する。(呪文)
            target_url = url.format(year,month)
            r = requests.get(target_url)
            soup = BeautifulSoup(r.content, 'html.parser') #天気は日本語で書いてある為、文字化け防止の為にr.textではなくr.contentとする。
            contents = soup.find_all('tr',class_='mtx',style='text-align:right;')
            sleep(1) #スクレイピングのマナーとして、1ページ読み込み毎に1秒待つ。
            for day in range(len(contents)):
                
                #種々の要素を読み込む。
                cday = (day+1) + (month*100) #日付
                Epressure = contents[day].find_all('td',class_='data_0_0')[0].text #現地気圧
                Opressure = contents[day].find_all('td',class_='data_0_0')[1].text #海面気圧
                precipitation = contents[day].find_all('td',class_='data_0_0')[2].text #合計降水量
                temperature = contents[day].find_all('td',class_='data_0_0')[5].text #平均気温
                humidity = contents[day].find_all('td',class_='data_0_0')[8].text #平均温度
                min_humidity = contents[day].find_all('td',class_='data_0_0')[9].text #最小湿度
                windspeed = contents[day].find_all('td',class_='data_0_0')[10].text #平均風速
                daylight = contents[day].find_all('td',class_='data_0_0')[15].text #日照時間
                snow = contents[day].find_all('td',class_='data_0_0')[16].text #合計降水量
                AMweather = contents[day].find_all('td',class_='data_0_0')[18].text #昼間天気
                PMweather = contents[day].find_all('td',class_='data_0_0')[19].text #夜間天気
                
                #スクレイピングしたデータをデータの種類毎に分けて辞書に保存する。
                c = {
                    '日':cday
                    ,'現地気圧':Epressure
                    ,'海面気圧':Opressure
                    ,'合計降水量':precipitation
                    ,'平均気温':temperature
                    ,'平均湿度':humidity
                    ,'最小湿度':min_humidity
                    ,'平均風速':windspeed
                    ,'日照時間':daylight
                    ,'合計降雪量':snow
                    ,'昼間天気':AMweather
                    ,'夜間天気':PMweather
                }
                
                #辞書のままリストに加えることで、データフレームにした時にキーの名前がそのままカラム名になってくれる。
                c_list.append(c)

            #スクレイピングの進捗を表示する。
            print("Scraping's progress:",year,"year",month,"month")
            print(target_url)

    #スクレイピングの終わりを宣言し、集めたデータをリスト型で返す。
    print("Scraping complited!")
    return c_list


#学習データを10年分スクレイピングする。
#CS(取得初年度,取得最終年度)
c_list = CS(2010,2019)

#スクレイピングした学習データをデータフレームにする。
c_df = pd.DataFrame(c_list)
display(c_df)


#テストデータを一年分スクレイピングする。
#startとendに同じ年を入れる事で、単年度のデータを取得できる。
c_t_list = CS(2020,2020)

#スクレイピングした学習データをデータフレームにする。
c_t_df = pd.DataFrame(c_t_list)
display(c_t_df)
