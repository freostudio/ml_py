#%%
import tushare as ts

df =ts.get_hist_data(code='600558',start='20200326',end='20200326',ktype='5',pause=3) #一次性获取全部日k线数据

# %%
df

# %%
df =ts.get_hist_data(code='600558',start='2020-03-24',end='2020-03-26',ktype='5',pause=3)

# %%
df.shape
# %%
import tushare as ts

df =ts.get_today_all()

# %%
df[df['code']=='600558']

# %%
import tushare as ts
ts.get_stock_basics().loc["300104"]

#%%
import tushare as ts

ts.get_latest_news() #默认获取最近80条新闻数据，只提供新闻类型、链接和标题
ts.get_latest_news(top=5,show_content=True) #显示最新5条新闻，并打印出新闻内容



# %%


import tushare as ts

df = ts.get_sina_dd('600558', date='2020-03-31') #默认400手
#df = ts.get_sina_dd('600848', date='2015-12-24', vol=500)  #指定大于等于500手的数据




# %%
df =df[['time','price','volume','type']]
# %%

df

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12),dpi=99)
##
plt.plot(df[df.type=='卖盘'].time,df[df.type=='卖盘'].volume,label='sale')
plt.plot(df[df.type=='买盘'].time,df[df.type=='买盘'].volume,label='buy')
plt.xlabel("time")
plt.ylabel("volume")
##plt.legend()
plt.show()

#%%
from datetime import datetime
xs = [datetime.strptime(d, '%H') for d in df[df.type=='卖盘'].time]

xs 

# %%
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
from datetime import datetime
##1-加载数据
df = ts.get_sina_dd('600558', date='2020-03-31') #默认400手
#df = ts.get_sina_dd('600848', date='2015-12-24', vol=500)  #指定大于等于500手的数据
##1-数据清洗处理-时间格式处理

df['time']=  pd.to_datetime(df['time'], format='%H:%M:%S')
df['time'] = df['time'].apply(lambda x:datetime.strftime(x,'%H:%M'))
df = df.sort_values(by = 'time',ascending=True)


#2-显示中文
#注意必须在差UN关键画布之前声明
plt.rcParams['font.sans-serif'] = 'SimHei'
#设置正常显示符号，解决保存图像是符号’-‘显示方块
plt.rcParams['axes.unicode_minus'] = False

#3-设置图形大小
plt.figure(figsize=(18,12),dpi=99)

#4-先显示所有数据
plt.scatter('time','volume',c='c',data=df,label ='中性')

# plt.scatter('time','volume',c='r',data=df[df.type=='卖盘'])
# plt.scatter('time','volume',c='c',data=df[df.type=='买盘'])

#5-根据type 加工需要特别显示的数据，用于两种数据同时显示
df_1 = {"time":df[df.type=='卖盘'].time,"volume":df[df.type=='卖盘'].volume}
df_2 = {"time":df[df.type=='买盘'].time,"volume":df[df.type=='买盘'].volume}
#转换为DF类型
df_1 =pd.DataFrame(df_1) 
df_2 =pd.DataFrame(df_2)
#6-显示多种类型数据不同颜色
plt.scatter('time','volume',c='r',data=df_1,label='卖出')
plt.scatter('time','volume',c='y',data=df_2,label='买入')

plt.xlabel("time")
plt.ylabel("volume")
plt.gcf().autofmt_xdate() # 自动旋转日期标记
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts
from datetime import datetime
##1-时间格式处理
df = ts.get_sina_dd('600558', date='2020-03-31') #默认400手
#df = ts.get_sina_dd('600848', date='2015-12-24', vol=500)  #指定大于等于500手的数据
df['time']=  pd.to_datetime(df['time'], format='%H:%M:%S')
df['time'] = df['time'].apply(lambda x:datetime.strftime(x,'%H-%M'))


# %%
df = df.sort_values(by = 'time',ascending=True)
df
# %%
##大盘行情
import tushare as ts

df = ts.get_index()
df

# %%

# %%
