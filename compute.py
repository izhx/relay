import math, random, time, json
from functools import wraps, reduce
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import csv
import copy
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 有中文出现的情况，需要u'内容'

# 输入
Sites = []  # 站的集合
SL = 0  # Sites length
# 计算出
_dis = np.zeros([2, 2], dtype=float)
# 辅助量
ConnList = []  # _dis<=20 的点集:[(i,j),(i,j)]
Outliers = []  # 孤立点集
Ones = np.ones(2, dtype=int)  # 全1的行向量
RS = np.zeros(2, dtype=int)
BF = np.zeros(2, dtype=int)


class Site(object):
    id = 0
    lng = 0  # 经度
    lat = 0  # 纬度
    kind = 0  # 0：RS，1：BF
    cluster = 0  # 哪个片区
    reachable = []  # 与当前点距离<=50的点集，才可能有连接
    reachList = []  # 上面的id和距离
    dis20_0 = []  # 距离20到0的点
    dis20_0List = []
    dis50_20 = []  # 距离为50到20的点集
    dis20_10 = []
    dis10_0 = []
    dis10_0List = []

    def __str__(self):
        return "site[" + str(self.id) + "], kind: " + str(self.kind)


class Solution(object):
    choose = np.zeros(2, dtype=int)  # 0是子站，1是宿主，要求的输出之一
    connect = np.zeros([2, 2], dtype=int)  # 1是两点之间有连接  有方向 i->j
    RR = np.zeros([2, 2], dtype=int)  # 1是子站与子站的连接关系
    DD = np.zeros([2, 2], dtype=int)  # 1是宿主与宿主
    DR = np.zeros([2, 2], dtype=int)  # 1是宿主与子站
    zone = np.zeros([2, 2], dtype=int)  # 都是宿主且统一片区时为1
    z = np.zeros([2, 2], dtype=int)  # 都是宿主且距离小于50时为1
    pl = np.zeros([2, 2], dtype=float)
    children = []
    hosts = []
    plan = {}  # 主站为key，{1:{"l1":[],"l2":[]}}
    son1 = []  # 所有的一级子站
    son2 = []
    son3 = []
    zones = []  # 片区二维数组
    can = False
    info = ""

    def __init__(self, len):
        self.choose = np.zeros(len, dtype=int)
        self.RR = np.zeros([len, len], dtype=int)
        self.DD = np.zeros([len, len], dtype=int)
        self.DR = np.zeros([len, len], dtype=int)
        self.connect = np.zeros([len, len], dtype=int)
        self.zone = np.zeros([len, len], dtype=int)
        self.z = np.zeros([len, len], dtype=int)
        self.pl = np.zeros([len, len], dtype=float)
        self.children = []
        self.hosts = []
        self.plan = {}
        self.son1 = []
        self.son2 = []
        self.son3 = []
        self.zones = []
        self.info = ""
        self.can = False


def func_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " 函数: {name} 开始执行...]".format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " 函数: {name} 执行完毕，耗时: {time:.2f}s]".format(name=function.__name__, time=t1 - t0))
        return result

    return function_timer


# @nb.vectorize("int64(int64)", nopython=True)
def sig(x):
    return 1 if x >= 0 else 0


# @nb.vectorize("int64(int64)", nopython=True)
def fan(x):  # 0-1 取反，矩阵就是用全1的减去就是反
    return 1 if x == 0 else 0


# 计算distance(i,j)矩阵
def distance(longitude1, latitude1, longitude2, latitude2):
    global _dis
    _dis = np.zeros([SL, SL], dtype=float)
    latitude1 = (math.pi / 180.0) * latitude1
    latitude2 = (math.pi / 180.0) * latitude2
    longitude1 = (math.pi / 180.0) * longitude1
    longitude2 = (math.pi / 180.0) * longitude2
    temp = np.sin(latitude1) * np.sin(latitude2) + \
           np.cos(latitude1) * np.cos(latitude2) * np.cos(longitude2 - longitude1)
    shit = np.where(temp > 1.0)  # 计算结果异常的点，由于精度损失可能会大于1
    temp[shit] = 1.0
    _dis = np.arccos(temp) * 6378.0
    _dis[(np.arange(0, SL), np.arange(0, SL))] = 0.0  # 由于精度损失，要设置对角线上全部为0
    return


# 生成每一个站点的几个临近点集
@func_timer
def generateSet():
    global Outliers
    global ConnList
    pos = np.where(_dis <= 20)
    for i, j in zip(pos[0], pos[1]):
        if i != j:
            ConnList.append((i, j))
    for i in range(0, SL):
        for j in range(0, SL):
            if i == j:
                continue
            else:
                if float(_dis[i, j]) > 20:
                    if float(_dis[i, j]) > 50:
                        continue
                    else:
                        Sites[i].dis50_20.append(j)
                else:
                    if float(_dis[i][j]) > 10:
                        Sites[i].dis20_10.append(j)
                    else:
                        Sites[i].dis10_0.append(j)
        Sites[i].dis20_0 = Sites[i].dis20_10 + Sites[i].dis10_0
        Sites[i].reachable = Sites[i].dis50_20 + Sites[i].dis20_0
        if len(Sites[i].reachable) == 0:
            Outliers.append(i)  # 如果它与其他所有点距离都大于50则为孤立点
        else:
            for r in Sites[i].reachable:
                Sites[i].reachList.append({"id": r, "dis": _dis[i, r]})
            Sites[i].reachList.sort(key=lambda x: x["dis"])
            for r in Sites[i].dis20_0:
                Sites[i].dis20_0List.append({"id": r, "dis": _dis[i, r]})
            Sites[i].dis20_0List.sort(key=lambda x: x["dis"])
            for r in Sites[i].dis10_0:
                Sites[i].dis10_0List.append({"id": r, "dis": _dis[i, r]})
            Sites[i].dis10_0List.sort(key=lambda x: x["dis"])

    return


# 画图
def drawSites():
    x = []
    y = []

    for i in range(0, SL):
        y.append(Sites[i].lng)
        x.append(Sites[i].lat)

    plt.scatter(x, y, s=10)  # 绘图
    plt.title(u'点', fontsize=14)  # 设置图表标题并给坐标轴加上标签
    plt.xlabel(u'经度', fontsize=14)
    plt.ylabel(u'纬度', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)  # 设置刻度标记的大小
    plt.axis([114, 118, 38, 40])  # 设置每个坐标轴的取值范围
    plt.show()
    return


def loadData(fileName):
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\" + fileName

    with open(path, 'r') as f:
        data = json.load(f)
        print("------已读取：", path)

    return data


def saveData():
    data = []

    for i in range(0, SL):
        data.append({"id": Sites[i].id, "lng": Sites[i].lng, "lat": Sites[i].lat,
                     "kind": Sites[i].kind})

    fileName = time.strftime("%m%d%H%M%S", time.localtime()) + ".json"
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\" + fileName

    with open(path, "w") as f:
        json.dump(data, f)
        print("------数据已保存: ", path)

    return fileName


def saveDis(dataName):
    fileName = dataName[:-5] + "_dis.csv"
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\" + fileName
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(0, SL):
            writer.writerow(_dis[i].tolist())
        print("------距离已保存: ", path)
    return fileName


def saveSolution(sol, dataName):
    data = {}

    data["dataName"] = dataName
    data["choose"] = sol.choose.tolist()
    data["hosts"] = sol.hosts
    data["children"] = sol.children
    data["can"] = sol.can
    data["info"] = sol.info
    data["son1"] = sol.son1
    data["son2"] = sol.son2
    data["son3"] = sol.son3
    data["plan"] = sol.plan
    data["zones"] = sol.zones

    fileName = time.strftime("%m%d%H%M%S_sol", time.localtime()) + ".json"
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\" + fileName
    with open(path, "w") as f:
        json.dump(data, f)
        print("------方案已保存: ", path)

    return fileName


def saveMat(zone, matName, solName):
    fileName = solName[:-5] + "_" + matName + ".csv"
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\" + fileName
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(0, SL):
            writer.writerow(zone[i].tolist())
        print("------矩阵已保存: ", path)
    return fileName


def saveAll(sol):
    dataName = saveData()
    saveDis(dataName)
    solName = saveSolution(sol, dataName)
    # saveMat(sol.zone, "zone", solName)
    # saveMat(sol.z, "z", solName)


# 初始化数据
@func_timer
def init(data):
    global Sites
    global Ones
    global SL
    global RS
    global BF
    SL = data  # len(data)
    Ones = np.ones(SL, dtype=int)
    RS = np.ones(SL, dtype=int)
    BF = np.zeros(SL, dtype=int)
    lat1 = np.zeros([SL, SL], dtype=float)
    lat2 = np.zeros([SL, SL], dtype=float)
    lng1 = np.zeros([SL, SL], dtype=float)
    lng2 = np.zeros([SL, SL], dtype=float)

    for i in range(0, SL):
        s = Site()
        s.id = i
        s.lat = random.uniform(112.9, 114.4)  # data[i]["lat"]
        lat1[i] = s.lat
        lat2[:, i] = s.lat
        s.lng = random.uniform(22.5, 23.4)  # data[i]["lng"]
        lng1[i] = s.lng
        lng2[:, i] = s.lng
        s.kind = random.randint(0, 1)  # data[i]["kind"]
        if s.kind:
            BF[i] = 1
        s.reachable = []
        s.reachList = []
        s.dis20_0 = []
        s.dis20_0List = []
        s.dis50_20 = []
        s.dis20_10 = []
        s.dis10_0 = []
        s.dis10_0List = []
        s.plan = {}
        Sites.append(s)

    RS = Ones - BF
    distance(lng1, lat1, lng2, lat2)
    generateSet()

    return


def checkConstraint(sol):
    # 约束1所用辅助量
    cdr = sol.connect * sol.DR  # 两矩阵对应位置元素相乘
    crr = sol.connect * sol.RR
    cdr_col_sum = np.sum(cdr, axis=0)  # axis为0，每一列元素相加，将矩阵压缩为一行
    crr_col_sum = np.sum(crr, axis=0)
    # 约束1 choose[i] == 1 时所用
    lk_1 = crr_col_sum + Ones  # 此处应用了numpy数组的广播
    kj_1 = np.dot(lk_1, crr) + Ones
    ji_1 = np.dot(kj_1, cdr)

    # 约束1 choose[i] == 0 时所用
    cdr_row_sum = np.sum(cdr, axis=1)  # axis为1，每一行元素相加，将矩阵压缩为一列
    # crr_row_sum = np.sum(crr, axis=1)
    jk_2 = np.sum(sol.connect * (sol.DR + sol.RR * cdr_row_sum), axis=1)
    ij_2 = np.sum(sol.connect * (sol.DR + sol.RR * jk_2), axis=1)

    # 约束2所用辅助量
    c_dis = sol.connect * _dis
    cdDR = c_dis * sol.DR
    cdDD = c_dis * sol.DD
    cdRR = c_dis * sol.RR

    # 约束1 建站
    # cons1_su_rs = (-cdr_col_sum + 4) * (-ji_1 + 6) * RS
    # cons1_su_bf = (-cdr_col_sum + 8) * (-ji_1 + 12) * BF
    # cons1_son = ij_2 * (Ones - sol.choose) + sol.choose
    # cons11 = np.where(cons1_su_rs < 0)[0]
    # if len(cons11):
    #     print("***cons1 suzhu rs failed:", cons11.__str__())
    #     return False
    # cons12 = np.where(cons1_su_bf < 0)[0]
    # if len(cons12):
    #     print("***cons1 suzhu bf failed:", cons12.__str__())
    #     return False
    # cons13 = np.where(cons1_son == 0)[0]
    # if len(cons13):
    #     print("***cons1 son failed:", cons13.__str__())
    #     return False

    for i in range(0, SL):  # 约束1 建站
        cons1 = 0
        if sol.choose[i]:
            if Sites[i].kind:
                cons1 = sig(8 - cdr_col_sum[i]) * sig(12 - ji_1[i])
            else:
                cons1 = sig(4 - cdr_col_sum[i]) * sig(6 - ji_1[i])
        else:
            cons1 = ij_2[i]
        if cons1:
            continue
        else:
            sol.info = "***cons1 failed. i=" + str(i)
            return False

    # 约束2 距离
    if len(np.where(cdDR > 20.0)[0]):
        sol.info = "***cdDR failed"
        return False
    elif len(np.where(cdDD > 50.0)[0]):
        sol.info = "***cdDD failed"
        return False
    elif len(np.where(cdRR > 10.0)[0]):
        sol.info = "***cdRR failed"
        return False
    else:
        pass

    sol.info = "Check Success!!!"
    return True


def setSite(sol, preset_hosts):
    sol.hosts = list(set(preset_hosts) | set(Outliers))

    for i in Outliers:  # 孤立点设置为宿主站
        sol.choose[i] = 1
        sol.plan[i] = {"sonNum": 0, "tree": []}

    cho = [n for n in range(0, SL) if n not in sol.hosts]  # 未设置的站点
    _preset_hosts = copy.deepcopy(preset_hosts)
    connect_preset(sol, cho, SL + 1, _preset_hosts)  # 对于没有选择建站方案的
    print("the rest cho : ", cho)
    while cho.__len__():
        sol.hosts.append(cho[0])
        sol.choose[cho[0]] = 1
        sol.plan[cho[0]] = {"sonNum": 0, "tree": []}
        cho.remove(cho[0])

    sol.children = [n for n in range(0, SL) if n not in sol.hosts]

    for i in range(0, SL):
        for j in Sites[i].reachable:
            sol.DD[i][j] = sol.choose[i] * sol.choose[j]
        for j in Sites[i].dis10_0:
            sol.RR[i][j] = fan(sol.choose[i]) * fan(sol.choose[j])
        for j in Sites[i].dis20_0:
            sol.DR[i][j] = fan(sol.DD[i][j]) * fan(sol.RR[i][j])

    if not checkConstraint(sol):
        print(sol.info)

    _create_tree(sol)
    return sol


def connect_preset(sol, cho, lengthBefore, preset_hosts):  # 按照最近原则
    length = len(cho)
    if length == 0:
        return
    elif length == lengthBefore:
        return
    else:
        random.shuffle(cho)

    while len(preset_hosts):  # 对于预设的宿主
        host = preset_hosts[0]
        sol.choose[host] = 1
        tree = []
        level1 = 8 if Sites[host].kind else 4
        near_set = set(np.where(_dis[host] < 40)[0])
        near_list = sorted(near_set, key=lambda x: _dis[host, x])
        may_l1_set = set(Sites[host].dis20_0) & set(cho)
        may_l1_list = sorted(may_l1_set, key=lambda x: _dis[host, x])
        edge_points = set()
        for s1 in may_l1_list:
            if s1 not in cho: continue
            sol.connect[s1, host] = 1
            cho.remove(s1)
            edge_points.add(s1)
            tree.append([host, s1])
            if len(tree) == level1: break
        l1_num = 0
        while l1_num != len(tree):
            may_l1_list.pop(0)
            l1_num += 1
        l23_num = (12 if Sites[host].kind else 6) - l1_num
        may_l23_list = near_list[l1_num:]  # 选过之后的那些
        for son in may_l23_list:
            if son not in cho: continue
            _may_dad_set = edge_points & set(Sites[son].dis10_0)
            _may_dad_list = sorted(_may_dad_set, key=lambda x: _dis[son, x])
            for _dad in _may_dad_list:
                for e in tree:
                    if len(e) > 3: continue
                    if e[-1] == _dad:
                        e.append(son)
                        break
                else:
                    continue
                sol.connect[son, _dad] = 1
                cho.remove(son)
                edge_points.remove(_dad)
                edge_points.add(son)
                l23_num -= 1
                break
            if not l23_num: break
        sol.plan[host] = {"sonNum": tree_site_num(tree), "tree": tree}
        preset_hosts.pop(0)

    if not len(preset_hosts):
        for i in cho:
            sol.choose[i] = 1
            cho.remove(i)
            l1 = []
            l2 = []
            l3 = []
            level1 = 8 if Sites[i].kind else 4
            level2 = 4 if Sites[i].kind else 2
            for s1 in Sites[i].dis20_0:  # 优先设置一级，
                if s1 not in cho:
                    continue
                else:
                    sol.connect[s1, i] = 1
                    cho.remove(s1)
                    l1.append({"dad": i, "id": s1})
                    sol.son1.append(s1)
                if len(l1) == level1: break
            for s1 in l1:  # 有1级再设置2级
                for s2 in Sites[s1["id"]].dis10_0:
                    if s2 not in cho:
                        continue
                    else:
                        sol.connect[s2, s1["id"]] = 1
                        cho.remove(s2)
                        l2.append({"dad": s1["id"], "id": s2})
                        sol.son2.append(s2)
                        break
                if len(l1) + len(l2) == level1 + level2: break
            else:
                for s2 in l2:  # 有2级再设置3级
                    for s3 in Sites[s2["id"]].dis10_0:
                        if s3 not in cho:
                            continue
                        else:
                            sol.connect[s3, s2["id"]] = 1
                            cho.remove(s3)
                            l3.append({"dad": s2["id"], "id": s3})
                            sol.son3.append(s3)
                            break
                    if len(l2) + len(l3) == level2: break

            sol.plan[i] = {"sonNum": len(l1) + len(l2) + len(l3), "l1": l1, "l2": l2, "l3": l3}

    return connect_preset(sol, cho, len(cho), preset_hosts)


# @func_timer
def chooseAndConnect(sol, cho, lengthBefore):
    length = len(cho)
    if length == 0:
        return
    elif length == lengthBefore:
        for i in cho:
            sol.choose[i] = 1
            sol.plan[i] = {"sonNum": 0, "l1": [], "l2": [], "l3": []}
            cho.remove(i)
        return
    else:
        random.shuffle(cho)
    # 对于没有选择建站方案的
    for i in cho:
        sol.choose[i] = 1
        cho.remove(i)
        l1 = []
        l2 = []
        l3 = []
        level1 = 4
        level2 = 2
        if Sites[i].kind:
            level1 = 8
            level2 = 4
        for s1 in Sites[i].dis20_0:  # 优先设置一级，
            if s1 not in cho:
                continue
            else:
                sol.connect[s1, i] = 1
                cho.remove(s1)
                l1.append({"dad": i, "id": s1})
                sol.son1.append(s1)
            if len(l1) == level1: break
        for s1 in l1:  # 有1级再设置2级
            for s2 in Sites[s1["id"]].dis10_0:
                if s2 not in cho:
                    continue
                else:
                    sol.connect[s2, s1["id"]] = 1
                    cho.remove(s2)
                    l2.append({"dad": s1["id"], "id": s2})
                    sol.son2.append(s2)
                    break
            if len(l1) + len(l2) == level1 + level2: break
        else:
            for s2 in l2:  # 有2级再设置3级
                for s3 in Sites[s2["id"]].dis10_0:
                    if s3 not in cho:
                        continue
                    else:
                        sol.connect[s3, s2["id"]] = 1
                        cho.remove(s3)
                        l3.append({"dad": s2["id"], "id": s3})
                        sol.son3.append(s3)
                        break
                if len(l2) + len(l3) == level2: break

        sol.plan[i] = {"sonNum": len(l1) + len(l2) + len(l3), "l1": l1, "l2": l2, "l3": l3}

    # print("CAC cho:", cho.__str__())
    return chooseAndConnect(sol, cho, length)


@func_timer
def planSite(sol):
    cho = [n for n in range(0, SL)]  # 未设置的站点

    for i in Outliers:  # 孤立点设置为宿主站
        sol.choose[i] = 1
        sol.plan[i] = {"sonNum": 0, "l1": [], "l2": [], "l3": []}
        cho.remove(i)

    chooseAndConnect(sol, cho, SL + 1)  # 对于没有选择建站方案的
    print("the rest cho : ", cho.__str__())
    while cho.__len__():
        sol.choose[cho[0]] = 1
        sol.plan[cho[0]] = {"sonNum": 0, "l1": [], "l2": [], "l3": []}
        cho.remove(cho[0])

    sol.hosts = []
    sol.children = []
    for i in range(0, SL):
        if sol.choose[i]:
            sol.hosts.append(i)
        else:
            sol.children.append(i)

    # pos = np.where(_dis > 20)
    # sol.connect[pos] = 0

    # _choose = Ones - sol.choose
    # sol.DD = sol.choose * sol.choose.reshape(-1, 1)
    # sol.RR = _choose * _choose.reshape(-1, 1)
    # sol.DR = (-sol.DD + 1) * (-sol.RR + 1)

    for i in range(0, SL):
        for j in Sites[i].reachable:
            sol.DD[i][j] = sol.choose[i] * sol.choose[j]
        for j in Sites[i].dis10_0:
            sol.RR[i][j] = fan(sol.choose[i]) * fan(sol.choose[j])
        for j in Sites[i].dis20_0:
            sol.DR[i][j] = fan(sol.DD[i][j]) * fan(sol.RR[i][j])

    return sol


def zoneNum(sol):  # 直接返回片区数量
    sol.zones = []
    flag = 0
    for i in sol.hosts:
        for z in sol.zones:
            for j in z:
                if j in Sites[i].reachable:
                    z.append(i)
                    flag = 1
                    break
            if flag: break
        else:
            sol.zones.append([i])

    return len(sol.zones)


@func_timer
def adjustZone(sol):  # 调整Zone
    ci = np.zeros([SL, SL], dtype=int)
    for i in range(0, SL):
        ci[i] = sol.choose[i]
        sol.z[i, i] = sol.choose[i]
    zm = _dis * ci * sol.choose
    pos = np.where((zm > 0) & (zm <= 50.0))
    sol.z[pos] = 1
    zt = sol.z + sol.z.T
    pos = np.where(zt >= 1)
    sol.z[pos] = 1

    l = 0
    nl = -1
    temp = sol.z.copy()
    while (l != nl):
        l = len(pos[0])
        temp = np.dot(temp, temp)
        pos = np.where(temp > 0)
        temp[pos] = 1
        nl = len(pos[0])

    sol.zone = temp
    return sol


@func_timer
def countZone(zone):
    # 列求和，压缩成一行，相同元素的个数/元素值
    z = {}
    num = 0
    zone_col_sum = np.sum(zone, axis=0)  # axis为0，每一列元素相加，将矩阵压缩为一行
    for i in range(0, SL):
        if zone_col_sum[i] in z:
            z[zone_col_sum[i]] += 1
        else:
            z[zone_col_sum[i]] = 1
    print(z)
    if 0 in z: del z[0]
    for key in z:
        num += z[key] / key
    return num


@func_timer
def getFeasibleSolution():  # 准备解
    sol = Solution(SL)

    while (sol.can == False):
        sol.can = checkConstraint(planSite(sol))
        print(sol.info)

    return sol


@func_timer
def getPresetSolution():
    sol = Solution(SL)
    preset_hosts = [n for n in range(SL) if random.randint(0, 1) and random.randint(0, 1) and random.randint(0, 1)]

    while not sol.can:
        sol.can = checkConstraint(setSite(sol, preset_hosts))
        print(sol.info)

    return sol


def optimize_solution(sol):
    _create_tree(sol)

    while True:
        old = copy.deepcopy(sol)
        old_number = evaluateSolution(old)
        old_score = old_number[0] + 3 * old_number[1]

        adjust_solution(sol)
        if checkConstraint(sol) == False:
            print(sol.info)
            continue

        now_number = evaluateSolution(sol)
        print(now_number)
        now_score = now_number[0] + 3 * now_number[1]

        if now_score >= old_score:
            break
        elif now_score / old_score > 0.9999:  # TODO
            return sol
        else:
            pass

    return old


@func_timer
def adjust_solution(sol):
    adjust_between_group(sol)
    adjust_in_group(sol)

    return sol


# 采用二维数组存树 [[0,1],[0,2,3],[0,4,5,6]] host:0, 一级：1 2 4，二级：3 5，三级：6
def _create_tree(sol):
    for host in sol.hosts:
        tree = []
        l1_list = list(np.where(sol.connect[:, host] == 1)[0])
        for l1 in l1_list:
            pos_l2 = np.where(sol.connect[:, l1] == 1)[0]
            if len(pos_l2):
                l2 = pos_l2[0]
                pos_l3 = np.where(sol.connect[:, l2] == 1)[0]
                if len(pos_l3):
                    tree.append([host, l1, l2, pos_l3[0]])
                else:
                    tree.append([host, l1, l2])
            else:
                tree.append([host, l1])
        sol.plan[host]["tree"] = tree

    return sol


def adjust_between_group(sol):
    for host in sol.hosts:
        for edge in sol.plan[host]["tree"]:
            point = edge[-1]
            d = _dis[host][point]
            pos = np.where(_dis[point] < d)[0].tolist()
            # 比自己宿主距离近的宿主 tar
            _host_set = set(pos).intersection(set(sol.hosts))
            if len(_host_set) == 0:
                continue
            else:
                # 找最近的一个host
                new_host = host
                for h in _host_set:
                    if _dis[point][h] < d:
                        d = _dis[point][h]
                        new_host = h
                # 找新宿主的边缘点离当前最近的  TODO 可以尝试直接加入
                d = _dis[host][point]
                new_edge = edge
                for n_edge in sol.plan[new_host]["tree"]:
                    if _dis[new_host][n_edge[-1]] < d:
                        d = _dis[new_host][n_edge[-1]]
                        new_edge = n_edge
                # 交换 满足10 或者20
                if new_edge != edge:
                    cons_old = 10 if len(edge) > 2 else 20  # 子站与子站10km，子站宿主20km
                    cons_new = 10 if len(new_edge) > 2 else 20
                    if _dis[new_edge[-1], edge[-2]] > cons_old:
                        continue
                    if _dis[edge[-1], new_edge[-2]] > cons_new:
                        continue
                    # print(edge)
                    sol.connect[edge[-1], edge[-2]] = 0
                    sol.connect[new_edge[-1], new_edge[-2]] = 0
                    sol.connect[edge[-1], new_edge[-2]] = 1
                    sol.connect[new_edge[-1], edge[-2]] = 1
                    new_edge[-1], edge[-1] = edge[-1], new_edge[-1]
                    # print(edge)

    return sol


# 组内调整  拓扑结构变换
def adjust_in_group(sol):
    for host in sol.hosts:
        if sol.plan[host]["sonNum"] == 0: continue  # todo 加入附近组
        old_tree = sol.plan[host]["tree"]
        new_tree = _change_tree(old_tree)
        if not new_tree: continue
        new_host = new_tree[0][0]
        if new_host != host:
            sol.plan[new_host] = {}
            sol.choose[host] = 0
            sol.choose[new_host] = 1
            sol.plan.pop(host)
        for oe in old_tree:
            for i in range(1, len(oe)):
                sol.connect[oe[i], oe[i - 1]] = 0
        for ne in new_tree:
            for i in range(1, len(ne)):
                sol.connect[ne[i], ne[i - 1]] = 1
        sol.plan[new_host]["sonNum"] = tree_site_num(new_tree)
        sol.plan[new_host]["tree"] = new_tree

    for i in range(0, SL):
        for j in Sites[i].reachable:
            sol.DD[i][j] = sol.choose[i] * sol.choose[j]
        for j in Sites[i].dis10_0:
            sol.RR[i][j] = fan(sol.choose[i]) * fan(sol.choose[j])
        for j in Sites[i].dis20_0:
            sol.DR[i][j] = fan(sol.DD[i][j]) * fan(sol.RR[i][j])

    return sol


def _change_tree(tree):
    # 选取中心几个点，分别当宿主，评价，最后选择一个方案
    # todo 优先 BF 宿主，因为12个子站
    sites = set()
    for edge in tree:
        sites = sites | set(edge)

    avg_lng = sum(map(lambda x: Sites[x].lng, sites)) / len(sites)
    avg_lat = sum(map(lambda x: Sites[x].lat, sites)) / len(sites)
    cen_dis = {}
    for s in sites:
        temp_dis = ((Sites[s].lng - avg_lng) ** 2 + (Sites[s].lat - avg_lat) ** 2) ** 0.5
        cen_dis[s] = temp_dis

    near_list = sorted(list(sites), key=lambda x: cen_dis[x])
    tree_dict = {}
    score_dict = {}
    for i in range(0, len(sites)):  # 全讨论
        temp_sites = copy.deepcopy(sites)
        temp_host = near_list[i]
        temp_sub_num = 12 if Sites[temp_host].kind else 6
        if temp_sub_num + 1 < len(sites): continue  # 如果临时宿主不能带起来小弟，pass掉
        temp_sites.remove(temp_host)
        temp_tree = _tree_init(temp_sites, temp_host)
        if not temp_tree: continue
        tree_dict[temp_host] = temp_tree
        score_dict[temp_host] = _tree_score(tree_dict[temp_host])

    score_list = sorted(score_dict, key=lambda x: score_dict[x])

    try:
        return tree_dict[score_list[0]]
    except:
        return False


# 生成比较好的树
def _tree_init(sites, host):
    site_set = copy.deepcopy(sites)
    level1 = 8 if Sites[host].kind else 4
    tree = []

    # 子站的类型: 1 只能1级 （10以内无站，距离host小于20）
    #            2 只能23级（10以内有站，距离host大于20）
    #            3 都可    （10以内有站，距离host小于20）
    #            4 连不上  （10以内无站，距离host大于20）
    can123 = {}
    can23 = {}
    l1_set = set()
    unset = copy.deepcopy(site_set)
    for sub in site_set:
        _within10 = set(Sites[sub].dis10_0) & unset
        if len(_within10) == 0:
            if _dis[sub, host] > 20:
                return False  # GG 有子站10km内无点且离host大于20，无效
            else:  # 只能1级
                tree.append([host, sub])
                l1_set.add(sub)
                unset.remove(sub)
        elif _dis[sub, host] > 20:
            can23[sub] = _within10  # 只能23
        else:
            can123[sub] = _within10  # 都可

    if len(tree) > level1: return False  # GG 一级子站超了

    ep_set = set() | l1_set
    free_l1_set = set() | l1_set
    temp_edge_list = []
    # 对于只能23的，先分配dad
    for sub in can23:  # 默认迭代key
        dad_set = (set(sorted(can23[sub], key=lambda x: _dis[sub, x]))) & unset
        if len(dad_set) == 0:
            return False  # todo GG
        for dad in dad_set & free_l1_set:
            # edge = list(filter(lambda x: x[1] == dad, tree))[0]
            edge = [e for e in tree if e[1] == dad][0]  # todo 是否正常引用了
            edge.append(sub)
            ep_set.add(sub)
            ep_set.remove(dad)
            free_l1_set.remove(dad)
            unset.remove(sub)
            break
        else:  #
            temp_dad_set = (dad_set - l1_set) & unset
            for _dad in temp_dad_set:
                _t_may1 = (free_l1_set | set(can123.keys())) & unset
                # _t_may1.remove(dad)
                _may1 = list(filter(lambda x: _dis[x, _dad] <= 10, _t_may1))
                if len(_may1):
                    if len(_may1) == 1:
                        tree.append([host, _may1[0], _dad, sub])
                        unset.remove(_may1[0])
                        unset.remove(_dad)
                        unset.remove(sub)
                        break
                    else:
                        temp_edge_list.append([_may1, _dad, sub])
                        unset.remove(_dad)
                        unset.remove(sub)
                        break
                elif _dis[_dad, host] > 20:
                    continue
            else:
                return False

    # 然后将temp_edge分配到1级
    _all_may1_set = set()
    for te in temp_edge_list:
        _con_set = set()
        for t2 in temp_edge_list:
            if t2 == te: continue
            if t2[0] == -1: continue  # todo -1换为[-1] ?
            try:
                _con_set = _con_set | set(set(te[0]) & set(t2[0]))
            except:
                pass
        if len(_con_set) < len(te[0]):
            _private = (set(te[0]) - _con_set) & unset
            for _p in _private:
                try:
                    tree.append([host, _p, te[1], te[2]])  # todo 现在是随机的
                    unset.remove(_p)
                    can123.pop(te[1])
                    break
                except:
                    pass

            te[0] = -1
    _times = 0

    for te in temp_edge_list:
        if te[0] != -1:
            _times = len(te[0]) if len(te[0]) > _times else _times
    for i in range(_times):
        temp_edge_list = [n for n in temp_edge_list if n[0] != -1]
        for te in temp_edge_list:
            _con_set = set()
            for t2 in temp_edge_list:
                if t2 == te: continue
                if te[0] == -1: continue  # todo -1换为[-1] ?
                _con_set = _con_set | set(set(te[0]) & set(t2[0]))
            if len(_con_set) < len(te[0]):
                _private = list(set(te[0]) - _con_set)
                tree.append([host, _private[0], te[1], te[2]])  # todo 现在是随机的
                try:
                    unset.remove(_private[0])
                    unset.remove(te[1])
                    can123.pop(te[1])
                except:
                    pass
                unset.remove(te[2])
                te[0] = -1

    if len(tree) > level1: return False
    if len(unset & set(can23.keys())): return False

    # todo can123 还没管  或许整个函数都动态更新，写dp
    print(can123.keys())
    print(unset)

    # 剩下的站按距离host由远及近排序，若有距离边缘点比host近的站，进行分配
    sub_list = sorted(can123.keys(), key=lambda x: _dis[host, x], reverse=True)
    may_edge_list = []
    for sub in sub_list:
        if sub not in unset: continue
        pos = set(np.where(_dis[sub] < _dis[sub, host])[0])
        may_dad_set = pos & can123[sub] & unset
        if len(may_dad_set):
            dad = sorted(may_dad_set, key=lambda x: _dis[sub, x])[0]
            may_edge_list.append([dad, sub])
            unset.remove(sub)
            unset.remove(dad)
        else:  # 只能为1级
            tree.append([host, sub])
            unset.remove(sub)
    for te in may_edge_list:
        pos = set(np.where(_dis[te[0]] < _dis[te[0], host])[0])
        may_dad_set = pos & can123[te[0]] & unset
        if len(may_dad_set):
            dad = sorted(may_dad_set, key=lambda x: _dis[sub, x])[0]
            tree.append([host, dad, te[0], te[1]])
            unset.remove(dad)
        else:  # 只能为1级
            tree.append([host, te[0], te[1]])

    if len(tree) > level1: return False  # GG 一级子站超了

    # # 剩下的站按距离host由远及近排序，若有距离边缘点比host近的站，进行分配
    # maybe23set = set(can123.keys()) | set(can23.keys())
    # maybe23list = sorted(maybe23set, key=lambda x: _dis[host, x], reverse=True)
    # for sub in maybe23list:
    #     pos = set(np.where(_dis[sub] < _dis[sub, host])[0])
    #     target_l1_set = pos & l1_set & set(Sites[sub].dis10_0)
    #     if len(target_l1_set) != 0:  # 如果这个sub附近有比host近的可连通的1级点
    #         dad = sorted(target_l1_set, key=lambda x: _dis[sub, x])[0]
    #         for e in tree:
    #             if e[1] == dad:
    #                 e.append(dad)
    #                 l1_set.remove(dad)
    #                 l1_set.add(sub)

    # subs = sorted(site_set, key=lambda x: _dis[host, x], reverse=True)

    # have_son = set()  # 在此的为两跳
    # for sub_index in range(len(subs)):
    #     sub_id = subs[sub_index]  # 当前站点id为 subs[sub_index]
    #     sub_edge = tree[sub_index]
    #     if len(sub_edge) > 3: continue  # 已经三跳   一跳[1,2]  二跳[1,2,3]  三跳[1,2,3,4]
    #
    #     # 得到比它离宿主近的sub列表，按照与它距离升序排列 todo 是否需要比他近，比它近利于组间调整但不是最优
    #     near_list = sorted(subs[sub_index + 1:], key=lambda x: _dis[sub_id, x])
    #     for near_sub in near_list:
    #         if near_sub in have_son: continue
    #         # 当前站到这个点n比到宿主近，回传损耗便少
    #         if _dis[sub_id, near_sub] < _dis[sub_id, host]:
    #             # 将sub_id 加入tree
    #             for edge in tree:
    #                 if edge[1] == near_sub:
    #                     edge.append(sub_id)
    #                     if sub_id in have_son:  # 此时为两跳  todo 考虑 20 10 的限制
    #                         edge.append(sub_edge[-1])
    #                     sub_edge[0] = -1
    #                     have_son.add(near_sub)
    #                     break
    #             break

    # index = 0
    # while index < len(tree):
    #     if tree[index][0] == -1:
    #         tree.pop(index)
    #     else:
    #         index += 1

    return tree


# todo 评分为 所有线段距离均值 再减去2*空位
def _tree_score(tree):
    vacancy_num = (12 if Sites[tree[0][0]].kind else 6) - tree_site_num(tree)
    score = vacancy_num * (-2)
    for edge in tree:
        score += sum(map(lambda x: _dis[edge[x - 1], edge[x]], range(1, len(edge))))
    return score


def tree_site_num(tree):
    num = 0
    for i in tree:
        num += len(i)
    return num - len(tree)


def sitesCost(sol):
    satellite = math.ceil(zoneNum(sol) / 8)
    return len(sol.hosts) * 10 + len(sol.children) * 5 + satellite * 50


# 计算i->j的路径损耗矩阵，返回系统平均损耗 TODO: 有warning
def pathLoss(sol):
    D = _dis * sol.connect * (sol.RR + sol.DR)
    sol.pl = (np.log10(D) + math.log10(900)) * 20 + 32.5
    pos = np.where(D == 0)
    sol.pl[pos] = 0.0
    return np.sum(sol.pl) / len(np.where(sol.connect == 1)[0])


def evaluateSolution(sol):
    ssc = sitesCost(sol)
    apl = pathLoss(sol)
    return (ssc, apl)


@func_timer
def finalOutput(sol):  # TODO: 处理并输出
    saveSolution(sol, saveData())
    return


def Convert2mine():
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\Result_1000_20_4.json"
    with open(path, 'r') as f:
        data = json.load(f)
        print("------已读取：", path)
    nodes = data[0]["nodes"]
    edges = data[0]["edges"]
    SL = 1000
    shit = [n for n in range(0, SL)]  # 未设置的站点
    for n in nodes:
        shit.remove(int(n["name"]) - 1)
    sol = Solution(SL)

    ss = []  # 存数据
    for i in range(0, SL):
        if i in shit:
            ss.append({"id": i, "lng": 0, "lat": 0, "kind": -1})
        else:
            ss.append({"id": i, "lng": 0, "lat": 0, "kind": 0})
    for n in nodes:
        id = int(n["name"]) - 1
        ss[id] = {"id": id, "lng": n["pos"][1] / (math.pi / 180.0),
                  "lat": n["pos"][0] / (math.pi / 180.0), "kind": 0}
        if n["tp"] == 1:
            sol.choose[id] = 1
            sol.hosts.append(id)
            sol.plan[id] = {"id": id, "l1": [], "l2": [], "l3": []}
        else:
            sol.children.append(id)
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\111.json"
    with open(path, "w") as f:
        json.dump(ss, f)
        print("------数据已保存: ", path)
    pl = 0

    # 先加一级
    so1 = {}  # key是子站id，value是父级id
    so2 = {}
    so3 = {}
    for e in edges:
        pl += (math.log10(float(e["dis"])) + math.log10(900)) * 20 + 32.5
        s = int(e["source"]) - 1
        t = int(e["target"]) - 1
        if s in sol.hosts:
            sol.plan[s]["l1"].append({"dad": s, "id": t})
            sol.son1.append(t)
            so1[t] = s
        elif t in sol.hosts:
            sol.plan[t]["l1"].append({"dad": t, "id": s})
            sol.son1.append(s)
            so1[s] = t

    for e in edges:
        s = int(e["source"]) - 1
        t = int(e["target"]) - 1
        if s in sol.hosts: continue
        if t in sol.hosts: continue
        if s in sol.son1:
            sol.plan[so1[s]]["l2"].append({"dad": s, "id": t})
            sol.son2.append(t)
            so2[t] = s
        elif t in sol.son1:
            sol.plan[so1[t]]["l2"].append({"dad": t, "id": s})
            sol.son2.append(s)
            so2[s] = t

    for e in edges:
        s = int(e["source"]) - 1
        t = int(e["target"]) - 1
        if s in sol.hosts: continue
        if t in sol.hosts: continue
        if t in so1: continue
        if s in so1: continue
        if s in sol.son2:
            sol.plan[so1[so2[s]]]["l3"].append({"dad": s, "id": t})
            sol.son3.append(t)
            so3[t] = s
        elif t in sol.son2:
            sol.plan[so1[so2[t]]]["l3"].append({"dad": t, "id": s})
            sol.son3.append(s)
            so3[s] = t

    sd = {}
    sd["dataName"] = "111.json"
    sd["choose"] = sol.choose.tolist()
    sd["hosts"] = sol.hosts
    sd["children"] = sol.children
    sd["son1"] = sol.son1
    sd["son2"] = sol.son2
    sd["son3"] = sol.son3
    sd["plan"] = sol.plan
    path = "C:\\izhx\\比赛\\数模\\2018深圳杯\\code\\wireless\\templates\\data\\111_sol.json"
    with open(path, "w") as f:
        json.dump(sd, f)
        print("------方案已保存: ", path)

    print("cost: ", len(sol.hosts) * 10 + len(sol.children) * 5)
    p = pl / len(edges)
    print("loss: ", p)

    pass


def prim(graph, vertex_num):
    INF = 1 << 20
    visit = [False] * vertex_num
    dist = [INF] * vertex_num
    preIndex = [0] * vertex_num
    for i in range(vertex_num):
        minDist = INF + 1
        nextIndex = -1
        for j in range(vertex_num):
            if dist[j] < minDist and not visit[j]:
                minDist = dist[j]
                nextIndex = j
        print(nextIndex)
        visit[nextIndex] = True
        for j in range(vertex_num):
            if dist[j] > graph[nextIndex][j] and not visit[j]:
                dist[j] = graph[nextIndex][j]
                preIndex[j] = nextIndex
    return dist, preIndex


if __name__ == "__main__":
    # print("** 当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # 初始化数据

    _ = 1 << 20
    graph = [
        [0, 6, 3, _, _, _],
        [6, 0, 2, 5, _, _],
        [3, 2, 0, 3, 4, _],
        [_, 5, 3, 0, 2, 3],
        [_, _, 4, 2, 0, 5],
        [_, _, _, 3, 5, 0],
    ]
    print(prim(graph, 6))

    init(1000)

    print('孤立点：', Outliers)

    # s = getFeasibleSolution()
    s = getPresetSolution()

    print(evaluateSolution(s))

    print(0)

    # optimize_solution(s)  todo

    # saveAll(s)

    # drawSites()

'''
0. 尽量把for改成numpy操作
1. 有规律的确定初始可行解
2. 尽可能短的有效编码，choose加上connect(i,j)去掉dis(i,j)>20的后线性化
3. 变异交叉等规则设定
'''
