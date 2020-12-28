import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import os.path as osp
import shutil
from tqdm import tqdm,trange
import math

# 配置参数
# 存储路径
save_path = 'pic/v0'
data_num = 5

# 中心船只的雷达半径，单位海里
radar_radius = 400

# 图中字体大小
fontsize = 15

# 画图会展示的列
display_cols = ['ShipName','ShipType','NavStatus','Length','Width','Draught']

# 各种实体的结构
config_dict = {
    '中心船舶':{
        'num_range':(1,2),
        'speed_range':(0,100),
        'draw_dict':{
            'markersize':'20',
            'fmt':'r*'
        }
    },
    '感知船舶':{
        'num_range':(2,7),
        'speed_range':(0,100),
        'draw_dict':{
            'markersize':'12',
            'fmt':'b^'
        }
    },
    '冰山':{
        'num_range':(0,2),
        'speed_range':(0,5),
        'draw_dict':{
            'markersize':'12',
            'fmt':'ro'
        },
        'Length_range':(50,100),
        'Width_range':(50,100),
        'Draught_range':(3,10)
    },
    '礁石':{
        'num_range':(0,2),
        'speed_range':(0,1),
        'draw_dict':{
            'markersize':'12',
            'fmt':'ro'
        },
        'Length_range':(50,100),
        'Width_range':(50,100),
        'Draught_range':(3,10)
    },
    '生物':{
        'num_range':(0,2),
        'speed_range':(0,10),
        'draw_dict':{
            'markersize':'12',
            'fmt':'ro'
        },
        'Length_range':(50,100),
        'Width_range':(50,100),
        'Draught_range':(3,10)
    }
}

# 配置边的生成参数
# 生成模拟关系格式
# 配置概率
# 船->船   护航：0.2，追捕：0.2，未知：0.6
# 船->非船 距离小于雷达半径的0.2 则避障 
ship2ship = {'护航':0.2,'追捕':0.4,'未知':1.0}
ship2noship = 0.2 
edge_columns = ['src_entity_id','relation_type','des_entity_id','src_DrawX','src_DrawY','des_DrawX','des_DrawY']
plt_relation_color_dict = {
    '护航':'g',
    '追捕':'b',
    '避障':'r'
}





# 通过米勒坐标系完成经纬度转换
class millerTransfer(object):
    @staticmethod
    def millerToXY (lon, lat):
        """
        经纬度(-180~180)转换为平面坐标系中的x,y 利用米勒坐标系
        :param lon: 经度
        :param lat: 维度
        :return:转换结果的单位是公里
        """
        L = 6381372*math.pi*2
        W = L
        H = L/2
        mill = 2.3
        x = lon*math.pi/180
        y = lat*math.pi/180
        y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))
        x = (W/2)+(W/(2*math.pi))*x
        y = (H/2)-(H/(2*mill))*y

        return x,y
    
    @staticmethod
    def millerToLonLat(x,y):
        """
        将平面坐标系中的x,y转换为经纬度，利用米勒坐标系
        :param x: x轴
        :param y: y轴
        :return:
        """
        L = 6381372 * math.pi*2
        W = L
        H = L/2
        mill = 2.3
        lat = ((H/2-y)*2*mill)/(1.25*H)
        lat = ((math.atan(math.exp(lat))-0.25*math.pi)*180)/(0.4*math.pi)
        lon = (x-W/2)*360/W

        return lon,lat



def load_test_data():
    """
    加载用于测试的船舶数据，测试船舶数据满足以下要求：
        1. ShipName,CallSign,ShipType,NavStatus,Length,Width,Draught
        2. 默认不存在的格式均为-1
    """
    dfp_test = pd.read_csv('data/船讯网数据样本_长江口数据.csv')
    dfp_test['ShipName'] = ['浙普渔运03',
     '浙普渔运03',
     '浙普渔运03',
     '浙普渔运03',
     '海巡01',
     '海巡01',
     '海巡01',
     '海巡01',
     '668588',
     '长安门号',
     '长安门号',
     '长安门号',
     '长安门号',
     '长安门号',
     '浙岭渔运08',
     '浙岭渔运08',
     '浙岭渔运08',
     '浙岭渔运08',
     '亨利04',
     '亨利04',
     '亨利04',
     '亨利04',
     '神华浚02',
     '神华浚02',
     '神华浚02',
     '神华浚02',
     '虎扑渔09',
     '虎扑渔09',
     '虎扑渔09',
     '虎扑渔09',
     '皇家创新号',
     '皇家创新号',
     '皇家创新号',
     '皇家创新号',
     '皇家创新号',
     '皇家创新号',
     '浙岱渔71',
     '浙岱渔71',
     '浙岱渔71',
     '浙岱渔71',
     '浙岱渔71',
     '浙岱渔71']
    dfp_use_test = dfp_test[['ShipName','CallSign','ShipTypeCN','NavStatusCN','Length','Width','Draught']]\
        .rename(columns={'ShipTypeCN':'ShipType','NavStatusCN':'NavStatus'})\
        .drop_duplicates()\
        .dropna(subset = ['ShipType'])\
        .reset_index(drop=True)
    
    return dfp_use_test

def generate_simulate_node(dfp_use):
    """
    输入:dfp_use，真实船舶数据
    """
    # 构建数据集
    columns = ['entity_id','entity_type','DrawX','DrawY',
               'Length','Width','Draught','Course',
               'Speed','ShipName','CallSign','ShipType',
               'NavStatus','Lon','Lat','display_text']

    dfp_list = []
    for entity_type in config_dict.keys():
        min_num,max_num = config_dict[entity_type]['num_range']
        num = np.random.randint(low=min_num,high=max_num)
#         print(f'生成{entity_type},数量:{num}')
        if num == 0:
            continue
        if entity_type in ['中心船舶','感知船舶']:
            dfp_sample = dfp_use.sample(n=num)
            # 不放回抽样
            dfp_use = dfp_use.iloc[list(set(range(len(dfp_use))) - set(dfp_sample.index)),:]
        else:
            # 非船舶实体，构造空列即可
            sample_data = {}
            for co in dfp_use.columns:
                if co in ['Length','Width','Draught']:
                    min_num,max_num = config_dict[entity_type][f'{co}_range']
                    sample_data[co] = np.random.rand(num)*(max_num-min_num) + min_num
                else:
                    sample_data[co] = [-1]*num
            dfp_sample = pd.DataFrame(sample_data)
        # 上entity_type,DrawX,DrwaY,Speed,Courese
        dfp_sample['entity_type'] = entity_type
        if entity_type == '中心船舶':
            dfp_sample['DrawX'] = 0.0
            dfp_sample['DrawY'] = 0.0
            middle_lon = np.random.rand(num).item() * 360 - 180
            middle_lat = np.random.rand(num).item() * 180 - 90
            middle_x,middle_y = millerTransfer.millerToXY(middle_lon,middle_lat)
            dfp_sample['Lon'] = middle_lon
            dfp_sample['Lat'] = middle_lat
        else:
            r = np.random.rand(num)*radar_radius
            theta = np.random.rand(num)*360
            dfp_sample['DrawX'] = r*np.cos(theta)
            dfp_sample['DrawY'] = r*np.sin(theta)
            # 计算其他目标的经纬度的步骤1.得到转化前的x,y 2.转化为经纬度
            dfp_middle = dfp_sample.apply(
                lambda s:millerTransfer.millerToLonLat(middle_x+s['DrawX'],middle_y+s['DrawY']),
                axis=1)
            dfp_sample['Lon'] = dfp_middle.apply(lambda s:s[0])
            dfp_sample['Lat'] = dfp_middle.apply(lambda s:s[1])


        min_num,max_num = config_dict[entity_type][f'speed_range']    
        dfp_sample['Speed'] = np.random.rand(num)*(max_num-min_num) + min_num
        dfp_sample['Course'] = np.random.rand(num)*360
        dfp_list.append(dfp_sample)

    dfp_node = pd.concat(dfp_list,axis=0).reset_index(drop=True)
    dfp_node['entity_id'] = range(dfp_node.shape[0])
    dfp_node['display_text'] = dfp_node.apply(
        lambda s:(f'船{s.entity_id}-' + '-'.join([str(s[co]) for co in display_cols])) if s.ShipName != -1 else f'{s.entity_type}{s.entity_id}',axis=1)
    dfp_node = dfp_node[columns]

    
    return dfp_node

def generate_simulate_edge(dfp_node):
    """
    生成模拟感知的边数据
    """
    def simulate_relation(s):
        if s.src_entity_id == s.des_entity_id:
            return '未知'

        if '船舶' in s.des_entity_type:
            r_num = np.random.rand(1).item()
            for re,value in ship2ship.items():
                if r_num < value:
                    return re
        else:
            # 船舶与非船舶的关系
            distance = math.sqrt((s.src_DrawX-s.des_DrawX)**2+
                              (s.src_DrawY-s.des_DrawY)**2)
            if distance < radar_radius*ship2noship:
                return '避障'
            else:
                return '未知'



    dfp_node_ship = dfp_node[dfp_node.entity_type.str.contains('船舶')][['entity_id','entity_type','DrawX','DrawY']]
    dfp_node_ship_src = dfp_node_ship.rename(columns=dict([(co,f'src_{co}') for co in dfp_node_ship.columns]))
    dfp_node_des = dfp_node[['entity_id','entity_type','DrawX','DrawY']]
    dfp_node_des = dfp_node_des.rename(columns=dict([(co,f'des_{co}') for co in dfp_node_ship.columns]))


    dfp_list = []
    for i,row_src in dfp_node_ship_src.iterrows():
        for j,row_des in dfp_node_des.iterrows():
            dfp_list.append(row_src.append(row_des))
    dfp_edge = pd.concat(dfp_list,axis=1).T
    dfp_edge['relation_type'] = dfp_edge.apply(simulate_relation,axis=1)
    dfp_edge = dfp_edge[dfp_edge.relation_type != '未知'][edge_columns]
    
    return dfp_edge
    
    

def plt_graph_without_relation(dfp_node):
    """
    生成不包含关系的态势图
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    # 先画雷达导航区域
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta) * radar_radius
    y = np.sin(theta) * radar_radius
    fig = plt.figure(figsize=(10, 10))
    # ax1 = plt.subplot(2,2,1)
    plt.plot(x, y, color="darkred", linewidth=2)

    #生成感知目标
    for index,row in dfp_node.iterrows():
        fmt = config_dict[row.entity_type]['draw_dict']['fmt']
        markersize = config_dict[row.entity_type]['draw_dict']['markersize']
        plt.plot(row.DrawX,row.DrawY,fmt,markersize=markersize,label=row.display_text)
        plt.text(row.DrawX,row.DrawY,row.display_text.split('-')[0]+'_'+f'{round(row.Speed)}节' ,fontsize = fontsize,alpha=0.8)  
        des_xy = (row.DrawX+np.sin(row.Course)*radar_radius/8,row.DrawY+np.cos(row.Course)*radar_radius/8)
        plt.annotate('',
                     xy=des_xy,
                     xytext=(row.DrawX,row.DrawY),
                     arrowprops=dict(arrowstyle="simple",connectionstyle="arc3")
                    )

    plt.legend(loc=2)
    plt.title('label:id-'+'-'.join(display_cols),fontsize=fontsize)
    
    return fig


def plt_graph_with_relation(dfp_node,dfp_edge):
    """
    生成包含关系的态势图
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    # 先画雷达导航区域
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta) * radar_radius
    y = np.sin(theta) * radar_radius
    fig = plt.figure(figsize=(10, 10))
    # ax1 = plt.subplot(2,2,1)
    plt.plot(x, y, color="darkred", linewidth=2)

    #生成感知目标
    for index,row in dfp_node.iterrows():
        fmt = config_dict[row.entity_type]['draw_dict']['fmt']
        markersize = config_dict[row.entity_type]['draw_dict']['markersize']
        plt.plot(row.DrawX,row.DrawY,fmt,markersize=markersize,label=row.display_text)
        plt.text(row.DrawX,row.DrawY,row.display_text.split('-')[0]+'_'+f'{round(row.Speed)}节' ,fontsize = fontsize,alpha=0.8)  
        des_xy = (row.DrawX+np.sin(row.Course)*radar_radius/8,row.DrawY+np.cos(row.Course)*radar_radius/8)
        plt.annotate('',
                     xy=des_xy,
                     xytext=(row.DrawX,row.DrawY),
                     arrowprops=dict(arrowstyle="simple",connectionstyle="arc3")
                    )

    # 生成态势感知，即关系
    for index,row in dfp_edge.iterrows():
        des_xy = (row.des_DrawX,row.des_DrawY)
        src_xy = (row.src_DrawX,row.src_DrawY)
        plt.annotate('',
                     xy=des_xy,
                     xytext=src_xy,
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc3",
                                     facecolor=plt_relation_color_dict[row.relation_type]),
                     alpha=0.5
        )
        text_x = (row.des_DrawX + row.src_DrawX)/2
        text_y = (row.des_DrawY + row.src_DrawY)/2
        plt.text(text_x,text_y,row.relation_type,fontsize=fontsize,color='r')

    plt.legend(loc=2)
    plt.title('label:id-'+'-'.join(display_cols),fontsize=fontsize)
    
    return fig
    
def generate_one_data(dfp_use):
    dfp_node = generate_simulate_node(dfp_use)
    dfp_edge = generate_simulate_edge(dfp_node)
    
    fig_without_relation = plt_graph_without_relation(dfp_node)
    fig_with_relation = plt_graph_with_relation(dfp_node,dfp_edge)
    
    return dfp_node,dfp_edge,fig_without_relation,fig_with_relation
    
def generate_num_data(num):
    
    if osp.exists(save_path):
        print('存储路径已经存在，删除后重新创建')
        shutil.rmtree(save_path)
        os.mkdir(save_path)   
    else:
        os.mkdir(save_path)   
    
    dfp_use = load_test_data()
    for i in trange(num,desc='生成模拟数据中'):
        name = str(i).zfill(4)
        dfp_node,dfp_edge,fig_without_relation,fig_with_relation = generate_one_data(dfp_use)
        save_path_i = osp.join(save_path,name)
        os.mkdir(save_path_i)
        
        dfp_node.to_csv(osp.join(save_path_i,f'node.csv'),index=False)
        dfp_edge.to_csv(osp.join(save_path_i,f'edge.csv'),index=False)
        fig_without_relation.savefig(osp.join(save_path_i,f'fig_without_relation.png'),
                                     dpi=500,bbox_inches = 'tight')
        fig_with_relation.savefig(osp.join(save_path_i,f'fig_with_relation.png'),
                                     dpi=500,bbox_inches = 'tight')
    print(f'生成{num}条模拟数据完成，路径{save_path}')
    
    return None
    
    
def main():
    """
    完成态势感知图的生成
    """
    generate_num_data(data_num)

if __name__ == "__main__":
    main()



