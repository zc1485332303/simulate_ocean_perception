"""
构建标注工具，用于标注模拟感知数据
"""
import streamlit as st
import pandas as pd
import os
import os.path as osp
from PIL import Image
import time
from main import load_test_data,generate_simulate_node,\
    generate_simulate_edge,plt_graph_without_relation,\
    plt_graph_with_relation


example_data_path = 'pic/v0/0000'
save_root = 'pic'
relation_type_dict = {
    0:'追捕',
    1:'护航',
    2:'避障'
}



def start_label(dfp_use,save_path):
    """
    ## 2.2 进入标注状态
    """
    try:
        data_id = max([int(i) for i in os.listdir(save_path) if i.isalnum()])
    except:
        data_id = -1
    data_id += 1
    number = st.number_input('点击+号进入下一文件的标注（第一次进行不需要点击+号,会默认寻找最大文件编号+1）',value=data_id,min_value=0,max_value=1000,step=1)
    data_id = str(number).zfill(4)
    f'**1. 当前标注文件id:{data_id}**'
    save_path_i = osp.join(save_path,data_id)
    if osp.exists(save_path_i):
        need2data_id = max([int(i) for i in os.listdir(save_path) if i.isalnum()]) + 1
        f'当前选择标注文件已经**存在**，不允许重复标注，请选择标注id:**{need2data_id}**'
    else:
        f'当前选择标注文件**不存在**，可以开始标注'
        f'**2. 生成模拟实体数据dfp_node**'
        dfp_node = generate_simulate_node(dfp_use)
        dfp_node
        f'**3. 生成无关系态势图**'
        fig_without_relation = plt_graph_without_relation(dfp_node)
        fig_without_relation
        """
        **4. 输入感知关系三元组**
        (src_entity_id,relation_type,des_entity_id)
        """
        st.write(f'当前实体id从{0,dfp_node.shape[0]}')
        st.write(relation_type_dict)
        # edge_str = st.text_input('输入关系标注,如示例使用,隔开', value='001,012,023', key=i+10000)
        edge_str = st.text_input('输入关系标注,如示例使用,隔开', value='001,012,023')
        edge_list = [[int(l[0]),relation_type_dict[int(l[1])],int(l[2])] for l in edge_str.split(',')]
        # edge_list
        dfp_edge = pd.DataFrame(edge_list,columns=['src_entity_id','relation_type','des_entity_id'])
        # dfp_edge
        # 获取可视化坐标
        dfp_edge_use = dfp_edge
        need_cols = ['entity_id','DrawX','DrawY']
        for s in ['src','des']:
            dfp_edge_use = dfp_edge_use.merge(dfp_node[need_cols],left_on=f'{s}_entity_id',right_on='entity_id',how='left')\
                .drop(columns=[need_cols[0]])\
                .rename(columns=dict([(co,f'{s}_{co}') for co in need_cols[1:]]))
        '**5. 生成模拟边数据dfp_edge**'
        dfp_edge
        '**6. 生成有关系态势图**'
        fig_with_relation = plt_graph_with_relation(dfp_node,dfp_edge_use)
        fig_with_relation

        # save = st.checkbox('标注完成请点击保存',key=i)
        save = st.button('标注完成请点击保存')
        if save:
            # 存储数据
            os.mkdir(save_path_i)
            dfp_node.to_csv(osp.join(save_path_i,f'node.csv'),index=False)
            dfp_edge.to_csv(osp.join(save_path_i,f'edge.csv'),index=False)
            fig_without_relation.savefig(osp.join(save_path_i,f'fig_without_relation.png'),
                                        dpi=500,bbox_inches = 'tight')
            fig_with_relation.savefig(osp.join(save_path_i,f'fig_with_relation.png'),
                                        dpi=500,bbox_inches = 'tight')
            '存储完毕,请回到2.2，选择开始下一个标注文件'
    
    return None


def main():
    st.title("无人艇环境感知模拟环境数据标注_v0")
    # number = st.number_input('Insert a number',value=0,min_value=0,max_value=1000,step=1)
    # st.write('The current number is ', number)
    # st.markdown("<font color=#0099ff size=7 face="黑体">color=#0099ff size=72 face="黑体"</font>")
    """
    # 1. 展示样例数据
    > 环境感知数据分别包含如下三块：
    * dfp_node:节点数据，包含态势感知中的各类实体（船舶和非船舶信息）
    * dfp_edge:边数据，包含态势感知中的各类实体之间的关系
    * fig_with_relation:dfp_node和dfp_edge可视化之后的结果
    """
    """
    ## 1.1 dfp_node
    表头 |含义 |数据类型 |说明
    ---|--- |--- |---
    entity_id| 实体id| int| 实体的唯一标识
    entity_type| 实体类型| str| 实体类型，比如船舶、礁石、冰山
    Length|长| int| 单位米
    Width|宽| int| 单位米
    Draught| 吃水| int| 单位米
    Course| 航向| int| 单位度，从正北方向顺时针旋转的角度
    Speed| 航速| float| 单位节/小时
    ShipName| 船名| str| 略
    CallSign| 呼号| str| 略
    ShipType| 船舶类型| str| 比如引航船,搜救船,捕捞,货船等
    NavStatus| 船舶状态| int| 比如航行，抛锚
    Lon| 经度| float| 单位度，-180～180,西经为负数
    Lat| 纬度| float| 单位度，-90～90,南纬为负数

    注：表中未出现但是数据中出现的列，属于画图的辅助列，不重要
    """
    dfp_node_example = pd.read_csv(osp.join(example_data_path,'node.csv'))
    dfp_node_example

    """
    1.2 dfp_edge

    表头 |含义 |数据类型 |说明
    ---|--- |--- |---
    src_entity_id| 开始实体id| int| 实体的唯一标识   
    relation_type| 关系类别| str| 比如避障，追捕，护航   
    des_entity_id| 目的实体id| int| 实体的唯一标识   

    注：表中未出现但是数据中出现的列，属于画图的辅助列，不重要
    """
    dfp_edge_example = pd.read_csv(osp.join(example_data_path,'edge.csv'))
    dfp_edge_example
    
    """
    1.3 fig_with_relation
    > 模拟态势感知图的可视化
    
    1.3.1 不带关系的图（标注前）
    """
    image = Image.open(osp.join(example_data_path,'fig_without_relation.png'))
    st.image(image,width=800)
    """
    1.3.2 带关系的图（标注后）
    """
    image = Image.open(osp.join(example_data_path,'fig_with_relation.png'))
    st.image(image,width=800)

    """
    # 2. 开始标注
    ## 2.1 创建标注文件夹
    """
    save_name = st.text_input('输入一个符合文件夹格式的名字', value='v0')
    save_path = osp.join(save_root,save_name)
    if not osp.exists(save_path):
        os.mkdir(save_path)
        f'创建新文件夹{save_name}存储标注数据'
    else:
        f'当前文件夹已经存在，继续使用'

    f'**当前标注所有结果存储路径为{save_path}**'
    dfp_use = load_test_data()
    f'**使用船舶数据库如下**'
    dfp_use[:5]


    start_label(dfp_use,save_path)


    


if __name__ == "__main__":
    main()