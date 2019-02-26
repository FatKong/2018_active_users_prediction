import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
from sklearn.feature_selection import RFE
import gc

def getset(x):
    return len(set(x))
def getkurt(x):
    return x.kurt()
def getvar(x):
    return x.var()
def getskew(x):
    return x.skew()
def App(app, usr):
    'app launch 表 特征'

    '2.登录次数'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'usr_launch_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.1用户登录次数/区间大小'
    secsize = app['day'].max() - app['day'].min()
    usr['launch_cnt_mean'] = usr['usr_launch_cnt'] / secsize


    '5.用户登录app的天数'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'launch_day_set']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '4.最后一天减最大日期'
    t = app
    launch_rank = t.groupby(by=['user_id']).rank(ascending=False)
    t['launch_rank_false'] = launch_rank
    t = t[t['launch_rank_false'] == 1]
    t.rename(columns={"day": 'last_launch_day'}, inplace=True)
    usr = pd.merge(usr, t[['user_id', 'last_launch_day']], on='user_id', how='left')
    ldm = usr['last_launch_day'].max()
    usr['uld_cut'] = ldm - usr['last_launch_day']
    usr = usr.drop(['last_launch_day'], axis=1)

    '6.用户启动的最大时间'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc='max').reset_index()
    t.columns = ['user_id', 'launch_day_max']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '7.用户启动的最小时间'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc='min').reset_index()
    t.columns = ['user_id', 'launch_day_min']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '7.1用户启动最大减最小时间'
    usr['launch_max_min'] = usr['launch_day_max'] - usr['launch_day_min']

    '7.2用户启动日期的中位数'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc=[np.median]).reset_index()
    t.columns = ['user_id', 'uld_median']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '7.3用户启动日期的平均数'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc='mean').reset_index()
    t.columns = ['user_id', 'uld_mean']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '7.4用户启动日期的峰度，方差，偏度'
    t = pd.pivot_table(app, index='user_id', values='day', aggfunc=[getkurt, getvar, getskew]).reset_index()
    t.columns = ['user_id', 'uld_kurt', 'uld_var', 'uld_skew']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '8.用户距离预测前两天登录次数'
    day_2 = app['day'].max() - 1
    t = app[app['day'] >= day_2]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_2_launch_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '9.用户距离预测前一天登录次数'
    day_1 = app['day'].max()
    t = app[app['day'] == day_1]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_1_launch_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '10.用户第一次登陆距离预测日的天数'
    pre_d = day_1 + 1
    usr['uld_first_cut'] = pre_d - usr['launch_day_min']

    '7.3.1用户登录时间的平均数中位数距离窗口末端的差'
    'uld_median'
    'uld_mean'
    usr['uld_median_cut'] = pre_d - usr['uld_median']
    usr['uld_mean'] = pre_d - usr['uld_mean']

    '11. 用户的每次登录时间距窗口末端的时间差特征 max(上面已经有) min mean var skw kurt'
    '11.1 用户每次登录时间距离窗口末端的时间差的均值'
    t=app
    t['day']=pre_d-t['day'] #统一减，减少运算量
    t=pd.pivot_table(t,index='user_id',values='day',\
    aggfunc=[np.mean,np.var,np.min, getskew,getkurt]).reset_index()
    t.columns=['user_id','launch_preday_avg','launch_preday_var','launch_preday_min','launch_preday_skw','launch_preday_kurt']
    usr=pd.merge(usr,t,on='user_id',how='left')

    '12.用户距离预测前3天的登录次数'
    day3 = app['day'].max() - 2
    t = app[app['day'] >= day3]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day3_launch_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    print('app done')
    return usr

    pass
def Act(act, usr):
    'act 表 特征---用户行为特征'


    '2.用户看视频的个数'
    t = pd.pivot_table(act, index='user_id', values='video_id', aggfunc='count').reset_index()
    t.columns = ['user_id', 'video_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.1.1用户发生行为的次数/区间大小'
    secsize = act['day'].max() - act['day'].min()
    usr['video_cnt_mean'] = usr['video_cnt'] / secsize

    '2.1用户发生行为的天数'
    t = pd.pivot_table(act, index='user_id', values='day', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'act_day_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.2用户平均每天看视频的个数'
    usr['video_cnt_per_day'] = usr['video_cnt'] / usr['act_day_cnt']

    '2.3用户看视频的种类'
    t = pd.pivot_table(act, index='user_id', values='video_id', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'video_set']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.4判别用户是否存在重复看视频'
    usr['if_video_2'] = usr[['video_cnt', 'video_set']].apply(lambda x: 1 if x['video_cnt'] != x['video_set'] else 0,
                                                              axis=1)

    '2.5用户重复看了多少个视频'
    usr['video_duplicates'] = usr[['video_cnt', 'video_set']].apply(lambda x: x['video_cnt'] - x['video_set'], axis=1)

    '2.6用户平均每天看视频的种类'
    usr['video_set_per_day'] = usr['video_set'] / usr['act_day_cnt']

    '3.用户看作者的个数'
    t = pd.pivot_table(act, index='user_id', values='author_id', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'author_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '3.1用户平均每天看作者的个数'
    usr['author_set_per_day'] = usr['author_cnt'] / usr['act_day_cnt']

    '4.用户在各类action type的次数'
    t = pd.pivot_table(act, index=['user_id', 'action_type'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'actype0', 'actype1', 'actype2', 'actype3', 'actype4', 'actype5']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '4.1用户在各类action type的比例'
    usr['actype0_ratio'] = usr['actype0'] / usr['video_cnt']
    usr['actype1_ratio'] = usr['actype1'] / usr['video_cnt']
    usr['actype2_ratio'] = usr['actype2'] / usr['video_cnt']
    usr['actype3_ratio'] = usr['actype3'] / usr['video_cnt']
    usr['actype4_ratio'] = usr['actype4'] / usr['video_cnt']
    usr['actype5_ratio'] = usr['actype5'] / usr['video_cnt']

    '5.用户在各个page发生次数'
    t = pd.pivot_table(act, index=['user_id', 'page'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'page0', 'page1', 'page2', 'page3', 'page4']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '5.1用户在各个page的比例'
    usr['page0_ratio'] = usr['page0'] / usr['video_cnt']
    usr['page1_ratio'] = usr['page1'] / usr['video_cnt']
    usr['page2_ratio'] = usr['page2'] / usr['video_cnt']
    usr['page3_ratio'] = usr['page3'] / usr['video_cnt']
    usr['page4_ratio'] = usr['page4'] / usr['video_cnt']

    '8.用户发生行为的最大 最小天数'
    t = pd.pivot_table(act, index='user_id', values='day', aggfunc=[np.max, np.min]).reset_index()
    t.columns = ['user_id', 'act_day_max', 'act_day_min']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '8.1用户行为日期最大-最小'
    usr['uad_max_min'] = usr['act_day_max'] - usr['act_day_min']

    '8.2用户第一次行为距离预测日的天数'
    pre_d = act['day'].max() + 1
    usr['uad_first_cut'] = pre_d - usr['act_day_min']

    '9.用户预测前两天行为次数'
    day_2 = act['day'].max() - 1
    t = act[['user_id', 'day']]
    t = t[t['day'] >= day_2]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_2_act_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '10.用户预测前1天的行为次数'
    day_1 = act['day'].max()
    t = act[['user_id', 'day']]
    t = t[t['day'] == day_1]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_1_act_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '10.1预测前一天看的视频种类getset'
    t = act[['user_id', 'day', 'video_id']]
    t = t[t['day'] == day_1]
    t = t[['user_id', 'video_id']]
    t = pd.pivot_table(t, index='user_id', values='video_id', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'day_1_video_set']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '10.2预测前一天看的作者数'
    t = act[['user_id', 'day', 'author_id']]
    t = t[t['day'] == day_1]
    t = t[['user_id', 'author_id']]
    t = pd.pivot_table(t, index='user_id', values='author_id', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_1_author_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '10.3预测前一天各个page的行为次数及比例'
    t = act[['user_id', 'day', 'page']]
    t = t[t['day'] == day_1]
    t = pd.pivot_table(t, index=['user_id', 'page'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'day_1_page0', 'day_1_page1', 'day_1_page2', 'day_1_page3', 'day_1_page4']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day_1_page0_ratio'] = usr['day_1_page0'] / usr['day_1_act_cnt']
    usr['day_1_page1_ratio'] = usr['day_1_page1'] / usr['day_1_act_cnt']
    usr['day_1_page2_ratio'] = usr['day_1_page2'] / usr['day_1_act_cnt']
    usr['day_1_page3_ratio'] = usr['day_1_page3'] / usr['day_1_act_cnt']
    usr['day_1_page4_ratio'] = usr['day_1_page4'] / usr['day_1_act_cnt']

    '10.4预测前一天各个actype的行为次数及比例'
    t = act[['user_id', 'day', 'action_type']]
    t = t[t['day'] == day_1]
    t = pd.pivot_table(t, index=['user_id', 'action_type'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'day_1_actype0', 'day_1_actype1', 'day_1_actype2', 'day_1_actype3', 'day_1_actype4',
                 'day_1_actype5']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day_1_actype0_ratio'] = usr['day_1_actype0'] / usr['day_1_act_cnt']
    usr['day_1_actype1_ratio'] = usr['day_1_actype1'] / usr['day_1_act_cnt']
    usr['day_1_actype2_ratio'] = usr['day_1_actype2'] / usr['day_1_act_cnt']
    usr['day_1_actype3_ratio'] = usr['day_1_actype3'] / usr['day_1_act_cnt']
    usr['day_1_actype4_ratio'] = usr['day_1_actype4'] / usr['day_1_act_cnt']
    usr['day_1_actype5_ratio'] = usr['day_1_actype5'] / usr['day_1_act_cnt']

    '11.用户行为日期的中位数，平均数，偏度，峰度，方差'
    t = pd.pivot_table(act, index='user_id', values='day',
                       aggfunc=[np.median, np.mean, getkurt, getvar, getskew]).reset_index()
    t.columns = ['user_id', 'act_day_median', 'act_day_mean', 'act_day_kurt', 'act_day_var', 'act_day_skew']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '12.用户行为日期到窗口末端的时间差的特征 max min mean var skew kurt'
    pre_d=act['day'].max()+1
    t=act[['user_id','day']]
    t['day']=pre_d-t['day']
    t=pd.pivot_table(t,index='user_id',values='day',\
    aggfunc=[np.max,np.mean,np.min,np.var,getskew,getkurt]).reset_index()
    t.columns=['user_id','act_preday_max','act_preday_avg','act_preday_min','act_preday_var','act_preday_skw','act_preday_kurt']
    usr=pd.merge(usr,t,on='user_id',how='left')

    '12.1 用户平均,最大，最小，中位数行为日期到窗口末端的差'
    usr['uad_mean_cut'] = pre_d - usr['act_day_mean']
    usr['uad_max_cut'] = pre_d - usr['act_day_max']
    usr['uad_min_cut'] = pre_d - usr['act_day_min']
    usr['uad_median_cut'] = pre_d - usr['act_day_median']

    '14.作者的action_type个数'
    t = pd.pivot_table(act, index=['author_id', 'action_type'], values='day', aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'authtype0', 'authtype1', 'authtype2', 'authtype3', 'authtype4', 'authtype5']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '15.作者被看次数'
    t = pd.pivot_table(act, index='author_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'bewatched_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '15.1作者的actype比率'
    usr['authtype0_ratio'] = usr['authtype0'] / usr['bewatched_cnt']
    usr['authtype1_ratio'] = usr['authtype1'] / usr['bewatched_cnt']
    usr['authtype2_ratio'] = usr['authtype2'] / usr['bewatched_cnt']
    usr['authtype3_ratio'] = usr['authtype3'] / usr['bewatched_cnt']
    usr['authtype4_ratio'] = usr['authtype4'] / usr['bewatched_cnt']
    usr['authtype5_ratio'] = usr['authtype5'] / usr['bewatched_cnt']

    '15.2作者被看的天数'
    t = pd.pivot_table(act, index='author_id', values='day', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'bewatched_day']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '15.3作者平均每天被看的次数'
    usr['bewatched_per_day'] = usr['bewatched_cnt'] / usr['bewatched_day']

    '15.4作者被看次数除以日期区间大小'
    usr['bewatched_cnt_mean'] = usr['bewatched_cnt'] / secsize

    '15.5作者在不同页面的行为次数及比例'
    t = pd.pivot_table(act, index=['author_id', 'page'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'authpage0', 'authpage1', 'authpage2', 'authpage3', 'authpage4']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['authpage0_ratio'] = usr['authpage0'] / usr['bewatched_cnt']
    usr['authpage1_ratio'] = usr['authpage1'] / usr['bewatched_cnt']
    usr['authpage2_ratio'] = usr['authpage2'] / usr['bewatched_cnt']
    usr['authpage3_ratio'] = usr['authpage3'] / usr['bewatched_cnt']
    usr['authpage4_ratio'] = usr['authpage4'] / usr['bewatched_cnt']

    '15.6作者被观看日期的最大最小平均中位数方差峰度偏度'
    t = pd.pivot_table(act, index='author_id', values='day',
                       aggfunc=[np.max, np.min, np.median, np.mean, np.var, getkurt, getskew]).reset_index()
    t.columns = ['user_id', 'authday_max', 'authday_min', 'authday_median', 'authday_avg', 'authday_var',
                 'authday_kurt', 'authday_skw']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '15.6.1作者被观看日期最大最小平均距离窗口末端的差'
    usr['authday_max_cut'] = pre_d - usr['authday_max']
    usr['authday_min_cut'] = pre_d - usr['authday_min']
    usr['authday_mean_cut'] = pre_d - usr['authday_avg']
    usr['authday_median_cut'] = pre_d - usr['authday_median']

    '15.6.2作者被观看日期距离窗口末端的时间差特征 max min mean median var kurt skew'
    t = act[['author_id', 'day']]
    t['day'] = pre_d - t['day']
    t = pd.pivot_table(t, index='author_id', values='day',
      aggfunc=[np.max, np.mean, np.min, np.var, getskew,getkurt]).reset_index()
    t.columns=['user_id','auth_preday_max','auth_preday_avg','auth_preday_min','auth_preday_var','auth_preday_skw', \
      'auth_preday_kurt']
    usr=pd.merge(usr,t,on='user_id',how='left')

    '16.用户所看视频的热度的统计特征'
    video_heat=pd.pivot_table(act,index='video_id',values='user_id',aggfunc=[len]).reset_index()
    video_heat.columns=['video_id','video_heat']
    tt=act[['user_id','video_id']]
    tt=pd.merge(tt,video_heat,on='video_id',how='left')
    t=pd.pivot_table(tt,index='user_id',values='video_heat',aggfunc=[np.max,np.min,np.mean,np.var,getskew,getkurt]).reset_index()
    t.columns=['user_id','video_heat_max','video_heat_min','video_heat_mean','video_heat_var','video_heat_skw','video_heat_kurt']
    usr=pd.merge(usr,t,on='user_id',how='left')

    '17.用户所看作者的热度统计特征'
    author_heat=pd.pivot_table(act,index='author_id',values='user_id',aggfunc=[len]).reset_index()
    author_heat.columns=['author_id','author_heat']
    tt=act[['user_id','author_id']]
    tt=pd.merge(tt,author_heat,on='author_id',how='left')
    t=pd.pivot_table(tt,index='user_id',values='author_heat',aggfunc=[np.max,np.min,np.mean,np.var,getskew,getkurt]).reset_index()
    t.columns=['user_id','author_heat_max','author_heat_min','author_heat_mean','author_heat_var','author_heat_skw','author_heat_kurt']
    usr=pd.merge(usr,t,on='user_id',how='left')

    """
       用户前三天的行为特征:
       1.行为次数 2.看视频种类 3.每天看视频个数/种类 4.重复看了多少个视频 5.用户看了多少个作者 6.每天看作者的个数
       7.在各个action_type个数/比例 8.action_type的七大统计特征 9.各个page的个数/比例 10.page的七大统计特征
       11.连续两天有行为的次数 12.日期的七大统计特征 13.最大最小平均中位数到窗口末端的差 14.到窗口末端时间差的七大统计特征
       15.用户关于日期间隔的六大特征
    """

    day3 = act['day'].max() - 2
    act = act[act['day'] >= day3]

    '1.行为次数即看视频个数'
    t = pd.pivot_table(act, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day3_video_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.看视频的种类'
    t = pd.pivot_table(act, index='user_id', values='video_id', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'day3_video_set']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '3.行为天数/每天看视频的个数/种类'
    t = pd.pivot_table(act, index='user_id', values='day', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'day3_dayset']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day3_video_pday'] = usr['day3_video_cnt'] / usr['day3_dayset']
    usr['day3_video_setpd'] = usr['day3_video_set'] / usr['day3_dayset']

    '4.重复看了多少视频'
    usr['day3_video_dup'] = usr[['day3_video_cnt', 'day3_video_set']].apply(lambda x: x['day3_video_cnt'] - x['day3_video_set'],axis=1)

    '5.用户看了多少个作者'
    t = pd.pivot_table(act, index='user_id', values='author_id', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'day3_author_set']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '6.每天看作者的个数'
    usr['day3_author_pd'] = usr['day3_author_set'] / usr['day3_dayset']

    '7.在各个action_type个数/比例'
    t = pd.pivot_table(act, index=['user_id', 'action_type'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'day3_actype0', 'day3_actype1', 'day3_actype2', 'day3_actype3', 'day3_actype4',
                 'day3_actype5']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day3_actype0_ratio'] = usr['day3_actype0'] / usr['day3_video_cnt']
    usr['day3_actype1_ratio'] = usr['day3_actype1'] / usr['day3_video_cnt']
    usr['day3_actype2_ratio'] = usr['day3_actype2'] / usr['day3_video_cnt']
    usr['day3_actype3_ratio'] = usr['day3_actype3'] / usr['day3_video_cnt']
    usr['day3_actype4_ratio'] = usr['day3_actype4'] / usr['day3_video_cnt']
    usr['day3_actype5_ratio'] = usr['day3_actype5'] / usr['day3_video_cnt']

    '9.各个page的个数/比例'
    t = pd.pivot_table(act, index=['user_id', 'page'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'day3_page0', 'day3_page1', 'day3_page2', 'day3_page3', 'day3_page4']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day3_page0_ratio'] = usr['day3_page0'] / usr['day3_video_cnt']
    usr['day3_page1_ratio'] = usr['day3_page1'] / usr['day3_video_cnt']
    usr['day3_page2_ratio'] = usr['day3_page2'] / usr['day3_video_cnt']
    usr['day3_page3_ratio'] = usr['day3_page3'] / usr['day3_video_cnt']
    usr['day3_page4_ratio'] = usr['day3_page4'] / usr['day3_video_cnt']

    '12.日期的七大统计特征'
    t = pd.pivot_table(act, index='user_id', values='day',
                       aggfunc=[np.max, np.min, np.median, np.mean, getvar, getkurt, getskew]).reset_index()
    t.columns = ['user_id', 'day3_max', 'day3_min', 'day3_median', 'day3_avg', 'day3_var',
                 'day3_kurt', 'day3_skw']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '13.最大最小平均中位数到窗口末端的差'
    pre_d = act['day'].max() + 1
    usr['day3_uad_mean_cut'] = pre_d - usr['day3_avg']
    usr['day3_uad_max_cut'] = pre_d - usr['day3_max']
    usr['day3_uad_min_cut'] = pre_d - usr['day3_min']
    usr['day3_uad_median_cut'] = pre_d - usr['day3_median']

    '14.到窗口末端时间差的七大统计特征'
    t = act[['user_id', 'day']]
    t['day'] = pre_d - t['day']
    t = pd.pivot_table(t, index='user_id', values='day',
                       aggfunc=[np.max, np.mean, np.min, np.var, getskew,getkurt]).reset_index()
    t.columns = ['user_id', 'day3_preday_max', 'day3_preday_avg', 'day3_preday_min', 'day3_preday_var',
                 'day3_preday_skw', 'day3_preday_kurt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    """
        作者前三天的行为特征：
        和上述用户的特征一样
        """

    '1.作者被看次数'
    t = pd.pivot_table(act, index='author_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day3_author_bewatched']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.作者被看的用户种类'
    t = pd.pivot_table(act, index='author_id', values='user_id', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'day3_bewtached_set']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '3.被看天数/每天被看个数/种类'
    t = pd.pivot_table(act, index='author_id', values='day', aggfunc=getset).reset_index()
    t.columns = ['user_id', 'day3_bewatched_dayset']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day3_bewatched_pd'] = usr['day3_author_bewatched'] / usr['day3_bewatched_dayset']
    usr['day3_bewatched_pdset'] = usr['day3_bewtached_set'] / usr['day3_bewatched_dayset']

    '4.作者重复被看次数'
    usr['day3_bewatched_dup'] = usr['day3_author_bewatched'] - usr['day3_bewtached_set']

    '5.6.没有'

    '7.在各个action_type个数/比例'
    t = pd.pivot_table(act, index=['author_id', 'action_type'], values='day', aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'day3_authtype0', 'day3_authtype1', 'day3_authtype2', 'day3_authtype3', 'day3_authtype4',
                 'day3_authtype5']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day3_authtype0_ratio'] = usr['day3_authtype0'] / usr['day3_author_bewatched']
    usr['day3_authtype1_ratio'] = usr['day3_authtype1'] / usr['day3_author_bewatched']
    usr['day3_authtype2_ratio'] = usr['day3_authtype2'] / usr['day3_author_bewatched']
    usr['day3_authtype3_ratio'] = usr['day3_authtype3'] / usr['day3_author_bewatched']
    usr['day3_authtype4_ratio'] = usr['day3_authtype4'] / usr['day3_author_bewatched']
    usr['day3_authtype5_ratio'] = usr['day3_authtype5'] / usr['day3_author_bewatched']

    '9.各个page的个数/比例'
    t = pd.pivot_table(act, index=['author_id', 'page'], values='day',
                       aggfunc='count').unstack().reset_index()
    t.columns = ['user_id', 'day3_authpage0', 'day3_authpage1', 'day3_authpage2', 'day3_authpage3', 'day3_authpage4']
    usr = pd.merge(usr, t, on='user_id', how='left')

    usr['day3_authpage0_ratio'] = usr['day3_authpage0'] / usr['day3_author_bewatched']
    usr['day3_authpage1_ratio'] = usr['day3_authpage1'] / usr['day3_author_bewatched']
    usr['day3_authpage2_ratio'] = usr['day3_authpage2'] / usr['day3_author_bewatched']
    usr['day3_authpage3_ratio'] = usr['day3_authpage3'] / usr['day3_author_bewatched']
    usr['day3_authpage4_ratio'] = usr['day3_authpage4'] / usr['day3_author_bewatched']

    '12.日期的七大统计特征'
    t = pd.pivot_table(act, index='author_id', values='day',
                       aggfunc=[np.max, np.min, np.median, np.mean, getvar, getkurt, getskew]).reset_index()
    t.columns = ['user_id', 'day3_authmax', 'day3_authmin', 'day3_authmedian', 'day3_authavg', 'day3_authvar',
                 'day3_authkurt', 'day3_authskw']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '13.最大最小平均中位数到窗口末端的差'
    usr['day3_auth_mean_cut'] = pre_d - usr['day3_authavg']
    usr['day3_auth_max_cut'] = pre_d - usr['day3_authmax']
    usr['day3_auth_min_cut'] = pre_d - usr['day3_authmin']
    usr['day3_auth_median_cut'] = pre_d - usr['day3_authmedian']

    '14.作者到窗口末端时间差的七大统计特征'
    t = act[['author_id', 'day']]
    t['day'] = pre_d - t['day']
    t = pd.pivot_table(t, index='author_id', values='day',
                       aggfunc=[np.max, np.mean, np.min, np.var, getskew,getkurt]).reset_index()
    t.columns = ['user_id', 'day3_authpreday_max', 'day3_authpreday_avg', 'day3_authpreday_min', 'day3_authpreday_var',
                 'day3_authpreday_skw', 'day3_authpreday_kurt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    'video --> user , author --> user , '
    'video, merge to user, user'


    print("act done")
    return usr
    pass
def Video(video,usr):
    'video 表 特征'

    '1.用户拍摄视频个数'
    t=pd.pivot_table(video,index='user_id',values='day',aggfunc='count').reset_index()
    t.columns=['user_id','mkvideo_cnt']
    usr=pd.merge(usr,t,on='user_id',how='left')

    '4.用户拍摄视频的 最大 最小 平均 方差 偏度 峰度 日期'
    t = pd.pivot_table(video, index='user_id', values='day', aggfunc=[np.max,np.min,np.mean,np.var,getkurt,getskew]).reset_index()
    t.columns = ['user_id', 'mk_max','mk_min','mk_mean','mk_var','mk_kurt','mk_skw']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '5.用户拍摄视频的最大最小时间差'
    usr['mk_max-min']=usr['mk_max']-usr['mk_min']

    '6.用户前三天拍摄视频的行为'
    '6.1用户最后一天拍摄的视频数'
    day_m = video['day'].max()
    t = video[video['day'] == day_m]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_1_mkvideo_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')

    '7.用户前三天的拍摄视频行为'
    day_3 = video['day'].max()-2
    t = video[video['day'] == day_3]
    t = pd.pivot_table(t, index='user_id', values='day', aggfunc='count').reset_index()
    t.columns = ['user_id', 'day_3_mkvideo_cnt']
    usr = pd.merge(usr, t, on='user_id', how='left')


    print('video done')
    return usr
    pass
def Reg(app, usr):

    'register 表 特征'
    '1.用户注册的来源渠道---该特征重要性低'
    t = pd.get_dummies(reg['reg_type'])
    t.columns = ['regtype0', 'regtype1', 'regtype2', 'regtype3', 'regtype4', 'regtype5', 'regtype6', 'regtype7',
                 'regtype8', \
                 'regtype9', 'regtype10', 'regtype11']
    t = pd.concat([reg[['user_id']], t], axis=1)
    usr = pd.merge(usr, t, on='user_id', how='left')

    '2.用户注册设备类型'
    usr = pd.merge(usr, reg[['user_id', 'dev_type']], on='user_id', how='left')

    '2.1用户注册设备类型离散化'
    usr['dev0'] = usr['dev_type'].map(lambda x: 1 if x == 0 else 0)
    t1 = [2, 4, 7, 8, 10, 1]
    usr['dev1'] = usr['dev_type'].map(lambda x: 1 if x in t else 0)
    t2 = [3, 15, 5, 6, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 34]
    usr['dev2'] = usr['dev_type'].map(lambda x: 1 if x in t2 else 0)
    t3 = [25, 35, 23, 22, 24, 26, 27, 28, 29, 30, 31, 33, 32, 37, 41, 36, 66, 47, 38, 45, 40, 44, 51, 39, 42, 43, 52,
          48, 50, 56, 46, 53, 55, 58, 49, 60, 59, 61, 57, \
          97, 54, 64, 65, 67, 70, 73, 62, 77, 74, 79, 69, 68, 83, 76, 72, 91, 63, 92, 84, 81, 71]
    usr['dev3'] = usr['dev_type'].map(lambda x: 1 if x in t3 else 0)
    usr['dev4'] = usr['dev_type'].map(lambda x: 1 if (x not in t1) & (x not in t2) & (x not in t3) else 0)

    '2.2该设备类型的注册数'
    t = reg[['user_id', 'dev_type']]
    t = pd.pivot_table(t, index='dev_type', values='user_id', aggfunc='count').reset_index()
    t.columns = ['dev_type', 'dev_type_cnt']
    usr = pd.merge(usr, t, on='dev_type', how='left')

    '3.用户注册时间与窗口末端的差值'
    usr = pd.merge(usr, reg[['user_id', 'reg_day']], on='user_id', how='left')
    day_fin = app['day'].max()
    usr['reg_day_fin'] = day_fin - usr['reg_day']

    '4.用户行为,登录平均日期到注册日期的差'
    usr['uad_mean_reg_cut'] = usr['act_day_mean'] - usr['reg_day']
    usr['uld_mean_reg_cut'] = usr['uld_mean'] - usr['reg_day']

    '5.用户行为，登录最大日期到注册日期的差'
    usr['uld_max_reg_cut'] = usr['launch_day_max'] - usr['reg_day']
    usr['uad_max_reg_cut'] = usr['act_day_max'] - usr['reg_day']

    print("reg done")
    return usr
    pass

	
	
def usersplit_test():
    '打标'
    print('label section.....')

    reg = pd.read_csv(r'..\user_register_log.txt', sep='\t', header=None)
    reg.columns = ['user_id', 'reg_day', 'reg_type', 'device_type']
    print('reg had read')

    app = pd.read_csv(r'..\app_launch_log.txt', sep='\t', header=None)
    app.columns = ['user_id', 'day']
    print('app had read')

    train_usr = reg[reg['reg_day'] <= 23]

    train_label = app[app['day'] >= 24]
    train_label = set(train_label['user_id'])

    print('start labeling')
    label = [1 if row['user_id'] in train_label else 0 for index, row in train_usr.iterrows()]
    train_usr['label'] = label

    print('label done')

    test_usr = reg

    return train_usr, test_usr
def usersplit_val():
    '打标'
    print('label section.....')

    reg = pd.read_csv(r'..\user_register_log.txt', sep='\t', header=None)
    reg.columns = ['user_id', 'reg_day', 'reg_type', 'device_type']
    print('reg had read')

    app = pd.read_csv(r'..\app_launch_log.txt', sep='\t', header=None)
    app.columns = ['user_id', 'day']
    print('app had read')


    train_usr = reg[reg['reg_day'] <= 23]

    train_label = app[app['day'] >= 24]
    train_label = set(train_label['user_id'])

    print('start labeling train')
    label = [1 if row['user_id'] in train_label else 0 for index, row in train_usr.iterrows()]
    train_usr['label'] = label

    val_usr=reg[reg['reg_day']<=20]

    val_label=app[(app['day']>=21)&(app['day']<=27)]
    val_label=set(val_label['user_id'])

    print('start labeling val')
    label = [1 if row['user_id'] in val_label else 0 for index, row in val_usr.iterrows()]
    val_usr['label']=label

    print('label done')

    return train_usr,val_usr

    pass

def ForTrain(train_usr):
    '训练集进程'

    'app划分'
    app = pd.read_csv(r'..\app_launch_log.txt', sep='\t', header=None)
    app.columns = ['user_id', 'day']

    train_app = app[(app.day >= 16) & (app.day <= 23)]
    train_usr = App(train_app, train_usr)

    'act'
    act = pd.read_csv(r'..\user_activity_log.txt', sep='\t', header=None)
    act.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']

    train_act = act[(act['day'] >= 16) & (act['day'] <= 23)]
    train_usr = Act(train_act, train_usr)

    'video'
    video=pd.read_csv(r'..\video_create_log.txt', sep='\t', header=None)
    video.columns=['user_id','day']

    train_video=video[(video['day']>=16)&(video['day']<=23)]
    train_usr=Video(train_video,train_usr)

    return train_usr
    pass
def ForVal(val_usr):
    'app划分'
    app = pd.read_csv(r'..\app_launch_log.txt', sep='\t', header=None)
    app.columns = ['user_id', 'day']

    val_app = app[(app.day >= 15) & (app.day <= 22)]
    val_usr = App(val_app, val_usr)

    'act'
    act = pd.read_csv(r'..\user_activity_log.txt', sep='\t', header=None)
    act.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']

    val_act = act[(act['day'] >= 15) & (act['day'] <= 22)]
    val_usr = Act(val_act, val_usr)

    'video'
    video = pd.read_csv(r'..\video_create_log.txt', sep='\t', header=None)
    video.columns = ['user_id', 'day']

    val_video = video[(video['day'] >= 15) & (video['day'] <= 22)]
    val_usr = Video(val_video, val_usr)

    return val_usr
    pass
def ForTest(test_usr):
    '测试集进程'

    'app划分'
    app = pd.read_csv(r'..\app_launch_log.txt', sep='\t', header=None)
    app.columns = ['user_id', 'day']

    test_app = app[app['day'] >= 24]
    test_usr = App(test_app, test_usr)

    'act'
    act = pd.read_csv(r'..\user_activity_log.txt', sep='\t', header=None)
    act.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']

    test_act = act[act['day'] >= 24]
    test_usr = Act(test_act, test_usr)

    'video'
    video = pd.read_csv(r'..\video_create_log.txt', sep='\t', header=None)
    video.columns = ['user_id', 'day']

    test_video = video[video['day'] >= 24]
    test_usr = Video(test_video, test_usr)

    return test_usr
    pass

def XgbVal(train,val):
    train = train.fillna(0)
    val = val.fillna(0)

    '训练前准备，去掉非特征列'
    train_y = train[['label']]
    train_x = train.drop(['user_id', 'label'], axis=1)

    val_y = val[['label']]
    val_x = val.drop(['user_id','label'], axis=1)

    xgb_train = xgb.DMatrix(train_x, label=train_y.values)
    xgb_val = xgb.DMatrix(val_x,label=val_y.values)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eta': '0.02',
              'max_depth': 5,
              'eval_metric': 'auc',
              # 'silent': 1,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18
              # 'max_delta_step':10
              }
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    model = xgb.train(params, xgb_train, evals=watchlist, num_boost_round=700)
    val_result = model.predict(xgb_val)


    auc=roc_auc_score(y_true=val_y.values,y_score=val_result)
    print(auc)
    plot_importance(model)
    plt.show()
    print('feat size', train_x.columns.size)
    pass
def XgbTest(train, test):
    train = train.fillna(0)
    test = test.fillna(0)

    '训练前准备，去掉非特征列'
    train_y = train[['label']]
    train_x = train.drop(['user_id', 'label'], axis=1)

    test_usr = test[['user_id']]
    test_x = test.drop(['user_id'], axis=1)

    xgb_train = xgb.DMatrix(train_x, label=train_y.values)
    xgb_test = xgb.DMatrix(test_x)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eta': '0.02',
              'max_depth': 5,
              'eval_metric': 'auc',
              # 'silent': 1,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 18
              # 'max_delta_step':10
              }
    model = xgb.train(params, xgb_train, num_boost_round=600)
    test_result = model.predict(xgb_test)

    test_usr['proba'] = test_result
    test_usr.to_csv('FatKong724.txt', encoding='utf-8', index=None, header=None)
    pass


def Mul_test():

    train_usr,test_usr=usersplit_test()

    p=Pool(2)
    Train=p.apply_async(ForTrain,args=[train_usr])
    Test=p.apply_async(ForTest,args=[test_usr])
    p.close()
    p.join()

    train=Train.get()
    test=Test.get()

    print('train', len(train))
    print('test', len(test))


    # train.to_csv(r'..\train.csv',index=False)
    # test.to_csv(r'..\test.csv',index=False)

    XgbTest(train=train,test=test)

    pass

def Mul_val():
    train_usr, val_usr = usersplit_val()

    p = Pool(2)
    Train = p.apply_async(ForTrain, args=[train_usr])
    Val = p.apply_async(ForVal, args=[val_usr])
    p.close()
    p.join()

    train = Train.get()
    val = Val.get()

    train.to_csv(r'..\train724.csv', index=False)
    val.to_csv(r'..\val724.csv', index=False)

    XgbVal(train=train, val=val)
    pass
