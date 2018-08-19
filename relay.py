import os
import time
import random
import math
import copy
import json
import csv
import xlrd
import numpy as np
from functools import wraps

# import multiprocessing
INF = 1 << 20


def func_timer(func):
    @wraps(func)
    def function_timer(*args, **kwargs):
        print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " 函数: {name} 开始执行...]".format(name=func.__name__))
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              " 函数: {name} 执行完毕，耗时: {time:.2f}s]".format(name=func.__name__, time=t1 - t0))
        return result

    return function_timer


def my_random():
    number = 0
    while number == 0:
        number = random.uniform(0, 1)
    return number


def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), 'Desktop')


def log(info=None):
    print("\n[", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "]\033[1;35m log \033[0m: ", info, "\n")


def sig(x):
    return 1 if x >= 0 else 0


def fan(x):  # 0-1 取反，矩阵就是用全1的减去就是反
    return 0 if x else 1


def tree_site_num(tree):
    num = 0
    for i in tree:
        num += len(i)
    return num - len(tree)


def get_random_data(size=1000, min_lat=112.9, max_lat=114.4, min_lng=22.5, max_lng=23.4):
    data = list()
    for i in range(size):
        data.append({"lat": random.uniform(min_lat, max_lat), "lng": random.uniform(min_lng, max_lng),
                     "kind": random.randint(0, 1), "preset": False})
    return data


#  余弦相似度，传入两个numpy行向量，算出余弦值
def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


#  jaccard系数，传入hosts
def get_jaccard_index(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


class Site(object):
    def __init__(self):
        self.id = -1
        self.lng = 0  # 经度
        self.lat = 0  # 纬度
        self.kind = -1  # 0：RS，1：BF
        self.problem_id = -1
        self.dis0_10 = []
        self.dis10_20 = []
        self.dis20_40 = []
        self.dis0_20 = []
        self.dis0_40 = []
        self.dis0_50 = []
        self.no = -1
        self.preset = False  # True为预设宿主站
        self.name = ""

    def str(self):
        return "site[" + str(self.id) + ": " + str(self.kind) + "]: lng:" + str(self.lng) + ", lat:" + str(self.lat)

    def has_anyone_within_50(self, site_id_list):
        intersection = set(self.dis0_50).intersection(set(site_id_list))
        return True if len(intersection) > 0 else False


class Solution(object):
    def __init__(self, length, problem_id):
        self.PROBLEM_ID = problem_id
        self.NUM = length
        self.SITES = Pre.SUB_PROBLEMS[problem_id].SITES
        self.DIS = Pre.SUB_PROBLEMS[problem_id].DIS
        self.choose = np.zeros(length, dtype=int)  # 0是子站，1是宿主，要求的输出之一
        self.connect = np.zeros([length, length], dtype=int)  # 1是两点之间有连接  有方向 i->j
        self.RR = np.zeros([length, length], dtype=int)  # 1是子站与子站的连接关系
        self.DD = np.zeros([length, length], dtype=int)  # 1是宿主与宿主
        self.DR = np.zeros([length, length], dtype=int)  # 1是宿主与子站
        self.zone = np.zeros([length, length], dtype=int)  # 都是宿主且统一片区时为1
        self.z = np.zeros([length, length], dtype=int)  # 都是宿主且距离小于50时为1
        self.pl = np.zeros([length, length], dtype=float)
        self.children = []
        self.hosts = []
        self.plan = {}
        self.zones = []  # 片区二维数组
        self.info = ""
        self.can = False
        self.avg_loss = 0
        self.sate_num = 0
        self.un_cover = 0
        self.all_cost = 0

    def show_tree(self, host_id):
        if host_id in self.hosts:
            for _edge in self.plan[host_id]["tree"]:
                for i in range(1, len(_edge)):
                    print("site: ", _edge[i - 1], "-", _edge[i], "dis: ", self.DIS[_edge[i - 1], _edge[i]])
        return

    def set_site(self, preset_hosts):
        _sites = Pre.SUB_PROBLEMS[self.PROBLEM_ID].SITES  # 引用
        # print("preset hosts", preset_hosts)
        unset = [n for n in range(0, self.NUM) if n not in preset_hosts]  # 未设置的站点
        self.connect_preset(unset, self.NUM + 1, preset_hosts)  # 对于没有选择建站方案的
        # print("the rest unset : ", unset)
        while unset.__len__():
            self.choose[unset[0]] = 1
            self.plan[unset[0]] = {"sonNum": 0, "tree": []}
            unset.remove(unset[0])

        for i in range(0, self.NUM):
            if self.choose[i] == 1:
                self.hosts.append(i)
            else:
                self.children.append(i)
            for j in _sites[i].dis0_50:
                self.DD[i][j] = self.choose[i] * self.choose[j]
            for j in _sites[i].dis0_10:
                self.RR[i][j] = fan(self.choose[i]) * fan(self.choose[j])
            for j in _sites[i].dis0_20:
                self.DR[i][j] = fan(self.DD[i][j]) * fan(self.RR[i][j])

        self.create_tree()
        self.create_zone(set(self.hosts))
        return

    def connect_preset(self, unset, length_before, preset_hosts):  # 按照最近原则  todo 待优化
        _sites = self.SITES
        length = len(unset)
        if length == 0:
            return
        elif length == length_before:
            return
        else:
            random.shuffle(unset)

        while len(preset_hosts):  # 对于预设的宿主
            host = preset_hosts[0]
            assert host not in unset  # 如果不是，证明逻辑出错
            preset_hosts.pop(0)
            self.choose[host] = 1
            tree = []
            level1 = 8 if _sites[host].kind == 1 else 4
            near_set = (set(_sites[host].dis0_40) - set(preset_hosts)) & set(unset)
            near_list = sorted(near_set, key=lambda x: self.DIS[host, x])
            may_l1_set = (set(_sites[host].dis0_20) - set(preset_hosts)) & set(unset)
            may_l1_list = sorted(may_l1_set, key=lambda x: self.DIS[host, x])
            edge_points = set()
            for s1 in may_l1_list:
                self.connect[s1, host] = 1
                unset.remove(s1)
                near_list.remove(s1)
                edge_points.add(s1)
                tree.append([host, s1])
                if len(tree) == level1:
                    break
            assert len(tree) <= level1  # 异常
            if len(tree) > level1:
                log(1)
            l23_num = (12 if _sites[host].kind == 1 else 6) - len(tree)
            for son in near_list:  # 选过之后的那些
                if son not in unset:
                    continue
                _may_dad_set = edge_points & set(_sites[son].dis0_10)
                _may_dad_list = sorted(_may_dad_set, key=lambda x: self.DIS[son, x])
                for _dad in _may_dad_list:
                    for e in tree:
                        if len(e) > 3:
                            continue
                        if e[-1] == _dad:
                            e.append(son)
                            self.connect[son, _dad] = 1
                            unset.remove(son)
                            edge_points.remove(_dad)
                            edge_points.add(son)
                            l23_num -= 1
                            break
                    else:
                        continue  # 到这里，对这个_dad啥也没做，继续下一个
                    break  # 没被continue就break掉，继续下一个son
                if l23_num == 0:
                    break
            _son_num = tree_site_num(tree)
            self.plan[host] = {"sonNum": _son_num, "kind": self.SITES[host].kind, "tree": tree}

        if len(preset_hosts) == 0:
            for i in unset:
                self.choose[i] = 1
                unset.remove(i)
                l1 = []
                l2 = []
                l3 = []
                level1 = 8 if _sites[i].kind else 4
                for s1 in _sites[i].dis0_20:  # 优先设置一级，
                    if s1 not in unset:
                        continue
                    else:
                        self.connect[s1, i] = 1
                        unset.remove(s1)
                        l1.append({"dad": i, "id": s1})
                    if len(l1) == level1:
                        break
                assert len(l1) <= level1
                level23 = (12 if _sites[i].kind else 6) - len(l1)
                for s1 in l1:  # 有1级再设置2级
                    for s2 in _sites[s1["id"]].dis0_10:
                        if s2 in unset:
                            self.connect[s2, s1["id"]] = 1
                            unset.remove(s2)
                            l2.append({"dad": s1["id"], "id": s2})
                            break
                    if len(l2) == level23:
                        break
                assert len(l2) <= level23
                if len(l2) < level23:
                    for s2 in l2:  # 有2级再设置3级
                        for s3 in _sites[s2["id"]].dis0_10:
                            if s3 in unset:
                                self.connect[s3, s2["id"]] = 1
                                unset.remove(s3)
                                l3.append({"dad": s2["id"], "id": s3})
                                break
                        if len(l2) + len(l3) == level23:
                            break
                assert len(l2) + len(l3) <= level23
                tree = []
                l1_list = list(np.where(self.connect[:, i] == 1)[0])
                for _l1 in l1_list:
                    pos_l2 = np.where(self.connect[:, _l1] == 1)[0]
                    if len(pos_l2):
                        _l2 = pos_l2[0]
                        pos_l3 = np.where(self.connect[:, _l2] == 1)[0]
                        if len(pos_l3):
                            tree.append([i, _l1, _l2, pos_l3[0]])
                        else:
                            tree.append([i, _l1, _l2])
                    else:
                        tree.append([i, _l1])
                self.plan[i] = {"sonNum": len(l1) + len(l2) + len(l3), "kind": self.SITES[i].kind, "tree": tree,
                                "l1": l1, "l2": l2, "l3": l3}

        return self.connect_preset(unset, length, preset_hosts)

    # 采用二维数组存树 [[0,1],[0,2,3],[0,4,5,6]] host:0, 一级：1 2 4，二级：3 5，三级：6
    def create_tree(self):
        for host in self.hosts:
            tree = []
            l1_list = list(np.where(self.connect[:, host] == 1)[0])
            for l1 in l1_list:
                pos_l2 = np.where(self.connect[:, l1] == 1)[0]
                if len(pos_l2):
                    l2 = pos_l2[0]
                    pos_l3 = np.where(self.connect[:, l2] == 1)[0]
                    if len(pos_l3):
                        tree.append([int(host), int(l1), int(l2), int(pos_l3[0])])
                    else:
                        tree.append([int(host), int(l1), int(l2)])
                else:
                    tree.append([int(host), int(l1)])
            self.plan[int(host)]["tree"] = tree

        return self

    def create_zone(self, unset):
        for h in self.hosts:
            if h not in unset:
                continue
            for z in self.zones:
                intersection = set(self.SITES[h].dis0_50) & set(z)
                if len(intersection) > 0:
                    z.append(h)
                    can_set = set(self.SITES[h].dis0_50) & unset
                    for i in can_set:
                        z.append(i)
                        unset.remove(i)
                    unset.remove(h)
                    break
            else:
                _z = list()
                _z.append(h)
                can_set = set(self.SITES[h].dis0_50) & unset
                for i in can_set:
                    _z.append(i)
                    unset.remove(i)
                self.zones.append(_z)

        while True:
            len_before = len(self.zones)
            change = False
            for index in range(1, len_before):
                for j in self.zones[index - 1]:
                    intersection = set(self.SITES[j].dis0_50) & set(self.zones[index])
                    if len(intersection) > 0:
                        new_z = list(set(self.zones[index - 1]) | set(self.zones[index]))
                        self.zones[index - 1] = new_z
                        self.zones.pop(index)
                        index -= 1
                        change = True
                        break
                if change:
                    break
            if len_before == len(self.zones):
                break
        return

    def satellite_num(self):  # todo
        # num = math.ceil(len(self.zones))  TODO zones里宿主总数会变少
        # if len(self.hosts) > 20:
        #     print(0)
        for z in self.zones:
            if len(z) == 1:
                self.un_cover += 1
            else:
                self.sate_num += math.ceil(len(z) / 8)

        num = self.sate_num
        return num

    def sites_cost(self):
        satellite = self.satellite_num()
        self.all_cost = len(self.hosts) * 10 + len(self.children) * 5 + satellite * 50
        return self.all_cost

    # 计算i->j的路径损耗矩阵，返回系统平均损耗  有warning
    def path_loss(self):
        d = self.DIS * self.connect * (self.RR + self.DR)
        self.pl = (np.log10(d) + math.log10(900)) * 20 + 32.5
        pos = np.where(d == 0)
        self.pl[pos] = 0.0
        self.avg_loss = np.sum(self.pl) / len(self.children)
        return self.avg_loss

    def evaluate(self):
        ssc = self.sites_cost()
        apl = self.path_loss()
        return ssc, apl

    def adjust_between_group(self):
        change = False
        for host in self.hosts:
            for edge in self.plan[host]["tree"]:
                point = edge[-1]  # 边缘点
                d = self.DIS[host][point]
                pos = list(map(lambda x: int(x), np.where(self.DIS[point] < d)[0].tolist()))
                # 比自己宿主距离近的宿主 tar
                _host_set = set(pos) & set(self.hosts)
                if len(_host_set) == 0:
                    continue
                else:
                    # 找最近的一个host
                    new_host = host
                    for h in _host_set:
                        if self.DIS[point][h] < d:
                            d = self.DIS[point][h]
                            new_host = h
                    # 找新宿主的边缘点离当前最近的  TODO 可以尝试直接加入
                    d = self.DIS[host][point]
                    new_edge = edge
                    for n_edge in self.plan[new_host]["tree"]:
                        if self.DIS[new_host][n_edge[-1]] < d:
                            d = self.DIS[new_host][n_edge[-1]]
                            new_edge = n_edge
                    # 交换 满足10 或者20
                    if new_edge != edge:
                        cons_old = 10 if len(edge) > 2 else 20  # 子站与子站10km，子站宿主20km
                        cons_new = 10 if len(new_edge) > 2 else 20
                        if self.DIS[new_edge[-1], edge[-2]] > cons_old:
                            continue
                        if self.DIS[edge[-1], new_edge[-2]] > cons_new:
                            continue
                        # print(edge)
                        self.connect[edge[-1], edge[-2]] = 0
                        self.connect[new_edge[-1], new_edge[-2]] = 0
                        self.connect[edge[-1], new_edge[-2]] = 1
                        self.connect[new_edge[-1], edge[-2]] = 1
                        new_edge[-1], edge[-1] = edge[-1], new_edge[-1]
                        change = True
                        # print(edge)
        return change

    def adjust_in_group(self):
        # 对于每一个小组
        change = False
        for host in self.hosts:
            for edge in self.plan[host]["tree"]:
                if len(edge) == 3:  # 一级  二级
                    l1 = edge[1]
                    l2 = edge[2]
                    if self.DIS[host, l2] < self.DIS[host, l1]:
                        self.connect[l1, host] = 0
                        self.connect[l2, host] = 1
                        self.connect[l2, l1] = 0
                        self.connect[l1, l2] = 1
                        edge[-2], edge[-1] = edge[-1], edge[-2]
                        change = True
                elif len(edge) == 4:
                    l1 = edge[1]
                    l2 = edge[2]
                    l3 = edge[3]
                    if self.DIS[l3, host] < self.DIS[l1, host]:
                        if self.DIS[l3, host] < self.DIS[l2, host]:
                            if self.DIS[l2, host] < self.DIS[l1, host]:
                                # 3 2 1 一定可以连
                                self.connect[l1, host] = 0
                                self.connect[l3, host] = 1
                                self.connect[l3, l2] = 0
                                self.connect[l2, l3] = 1
                                self.connect[l2, l1] = 0
                                self.connect[l1, l2] = 1
                                edge[1], edge[3] = edge[3], edge[1]
                                change = True
                            elif self.DIS[l3, l1] <= 10:  # 3 1 2
                                self.connect[l1, host] = 0
                                self.connect[l3, host] = 1
                                self.connect[l3, l2] = 0
                                self.connect[l1, l3] = 1
                                edge[1], edge[3] = edge[3], edge[1]
                                edge[3], edge[2] = edge[2], edge[3]
                                change = True

                    elif self.DIS[l2, host] < self.DIS[l1, host]:
                        if self.DIS[l1, l3] <= 10:
                            old_path = self.DIS[host, l1] + self.DIS[l2, l1] + self.DIS[l2, l3]
                            new_path = self.DIS[host, l2] + self.DIS[l2, l1] + self.DIS[l1, l3]
                            if new_path < old_path:
                                # change  2 1 3
                                self.connect[l1, host] = 0
                                self.connect[l2, host] = 1
                                self.connect[l2, l1] = 0
                                self.connect[l1, l2] = 1
                                self.connect[l3, l2] = 0
                                self.connect[l3, l1] = 1
                                edge[-2], edge[-1] = edge[-1], edge[-2]
                                change = True

        return change


class SubProblem(object):
    def __init__(self):
        self.id = -1
        self.NUM = 0
        self.SITES = []
        self.site_id_list = []
        self.DIS = np.zeros([self.NUM, self.NUM], dtype=float)
        self.RS = np.ones(self.NUM, dtype=int)
        self.BF = np.zeros(self.NUM, dtype=int)
        self.ONES = np.ones(self.NUM, dtype=int)  # 全1的行向量
        self.preset_hosts = set()

    def has_site(self, site):
        # return True if site.id in self.site_id_list else False
        if site.id in self.site_id_list:
            index = self.site_id_list.index(site.id)
            if site != self.SITES[index]:
                log("当前site(" + str(site.id) + ")在此area(" + str(self.id) + ")列表中但object不同！")
                return False
            else:
                return True
        else:
            return False

    def put_site(self, site):
        site.no = len(self.site_id_list)
        self.site_id_list.append(site.id)
        self.SITES.append(site)
        site.problem_id = self.id  # 利用python对象引用直接改，ALL_SITES也会跟着变

    def can_put(self, site):
        intersection = set(site.dis0_50) & set(self.site_id_list)
        return True if len(intersection) > 0 else False

    def init(self):
        # 生成各个矩阵
        # 评价每个站点，是否适合当host sub
        # 每个group能自动调整自身形态到最低损耗
        self.NUM = len(self.SITES)
        self.BF = np.zeros(self.NUM, dtype=int)
        _lat1 = np.zeros([self.NUM, self.NUM], dtype=float)
        _lat2 = np.zeros([self.NUM, self.NUM], dtype=float)
        _lng1 = np.zeros([self.NUM, self.NUM], dtype=float)
        _lng2 = np.zeros([self.NUM, self.NUM], dtype=float)

        for i in range(0, self.NUM):
            self.SITES[i].no = i
            _lng1[i] = self.SITES[i].lng
            _lng2[:, i] = self.SITES[i].lng
            _lat1[i] = self.SITES[i].lat
            _lat2[:, i] = self.SITES[i].lat
            if self.SITES[i].kind == 1:
                self.BF[i] = 1
            if self.SITES[i].preset:
                self.preset_hosts.add(i)

        self.ONES = np.ones(self.NUM, dtype=int)  # 全1的行向量
        self.RS = self.ONES - self.BF
        self.DIS = Pre.distance(self.NUM, _lng1, _lat1, _lng2, _lat2)

        for site in self.SITES:
            _dis0_10 = np.where(self.DIS[site.no] <= 10)[0].tolist()
            _dis0_10.remove(site.no)  # .remove(site.id)  会变成noneType
            site.dis0_10 = sorted(_dis0_10, key=lambda x: self.DIS[site.no, x])
            _dis0_20 = np.where(self.DIS[site.no] <= 20)[0].tolist()
            _dis0_20.remove(site.no)
            site.dis0_20 = sorted(_dis0_20, key=lambda x: self.DIS[site.no, x])
            _dis0_40 = np.where(self.DIS[site.no] <= 40)[0].tolist()
            _dis0_40.remove(site.no)
            site.dis0_40 = sorted(_dis0_40, key=lambda x: self.DIS[site.no, x])
            _dis0_50 = np.where(self.DIS[site.no] <= 50)[0].tolist()
            _dis0_50.remove(site.no)
            site.dis0_50 = sorted(_dis0_50, key=lambda x: self.DIS[site.no, x])
            _dis10_20 = set(site.dis0_20) - set(site.dis0_10)
            site.dis10_20 = sorted(_dis10_20, key=lambda x: self.DIS[site.no, x])
            _dis20_40 = set(site.dis0_40) - set(site.dis0_20)
            site.dis10_20 = sorted(_dis20_40, key=lambda x: self.DIS[site.no, x])

        return

    def check_constraint(self, solution):
        # 约束1所用辅助量
        cdr = solution.connect * solution.DR  # 两矩阵对应位置元素相乘
        crr = solution.connect * solution.RR
        cdr_col_sum = np.sum(cdr, axis=0)  # axis为0，每一列元素相加，将矩阵压缩为一行
        crr_col_sum = np.sum(crr, axis=0)
        # 约束1 choose[i] == 1 时所用
        lk_1 = crr_col_sum + self.ONES  # 此处应用了numpy数组的广播
        kj_1 = np.dot(lk_1, crr) + self.ONES
        ji_1 = np.dot(kj_1, cdr)
        # 约束1 choose[i] == 0 时所用
        cdr_row_sum = np.sum(cdr, axis=1)  # axis为1，每一行元素相加，将矩阵压缩为一列
        jk_2 = np.sum(solution.connect * (solution.DR + solution.RR * cdr_row_sum), axis=1)
        ij_2 = np.sum(solution.connect * (solution.DR + solution.RR * jk_2), axis=1)
        # 约束2所用辅助量
        c_dis = solution.connect * self.DIS
        cd_dr = c_dis * solution.DR
        cd_dd = c_dis * solution.DD
        cd_rr = c_dis * solution.RR

        for i in range(0, self.NUM):  # 约束1 建站
            if solution.choose[i]:
                if self.SITES[i].kind:
                    cons1 = sig(8 - cdr_col_sum[i]) * sig(12 - ji_1[i])
                else:
                    cons1 = sig(4 - cdr_col_sum[i]) * sig(6 - ji_1[i])
            else:
                cons1 = ij_2[i]
            if cons1:
                continue
            else:
                solution.info = "***cons1 failed. i=" + str(i)
                log(solution.info)
                return False

        # 约束2 距离
        if len(np.where(cd_dr > 20.0)[0]):
            solution.info = "***cd_dr failed"
            return False
        elif len(np.where(cd_dd > 50.0)[0]):
            solution.info = "***cd_dd failed"
            return False
        elif len(np.where(cd_rr > 10.0)[0]):
            solution.info = "***cd_rr failed"
            return False
        else:
            pass

        solution.info = "Check Success!"
        return True

    def get_preset_solution(self, preset_hosts=None):
        _sol = Solution(self.NUM, self.id)
        if preset_hosts is None:
            _preset_hosts = list(self.preset_hosts)
        else:
            _preset_hosts = list(self.preset_hosts | set(preset_hosts))

        while not _sol.can:
            # print(_sol.info)
            random.shuffle(_preset_hosts)
            _sol.set_site(_preset_hosts)
            _sol.can = self.check_constraint(_sol)

        return _sol


class Pre(object):
    ALL_SITES = []
    ALL_NUM = 0
    PRE_DIS = np.zeros([2, 2], dtype=float)
    SUB_PROBLEMS = []  # 片区数组 一个孤立点算一个片区，计算卫星时要有
    CONNECT = np.zeros([2, 2], dtype=float)
    CHOOSE = np.zeros([2, 2], dtype=float)

    @classmethod
    @func_timer
    def process_data(cls, data):
        cls.SUB_PROBLEMS = list()
        cls.ALL_SITES = list()
        cls.ALL_NUM = len(data)
        cls.PRE_DIS = np.zeros([cls.ALL_NUM, cls.ALL_NUM], dtype=float)
        lat1 = np.zeros([cls.ALL_NUM, cls.ALL_NUM], dtype=float)
        lat2 = np.zeros([cls.ALL_NUM, cls.ALL_NUM], dtype=float)
        lng1 = np.zeros([cls.ALL_NUM, cls.ALL_NUM], dtype=float)
        lng2 = np.zeros([cls.ALL_NUM, cls.ALL_NUM], dtype=float)

        for i in range(0, cls.ALL_NUM):
            site = Site()
            site.id = i
            site.name = data[i]["site_name"]
            site.lat = data[i]["lat"]
            lat1[i] = site.lat
            lat2[:, i] = site.lat
            site.lng = data[i]["lng"]
            lng1[i] = site.lng
            lng2[:, i] = site.lng
            site.kind = data[i]["kind"]
            site.preset = data[i]["preset"]
            cls.ALL_SITES.append(site)
            del site

        cls.PRE_DIS = cls.distance(cls.ALL_NUM, lng1, lat1, lng2, lat2)
        for site in cls.ALL_SITES:
            site.dis0_50 = np.where(cls.PRE_DIS[site.id] <= 50)[0].tolist()  # .remove(site.id)  会变成noneType
            site.dis0_50.remove(site.id)

        cls.divide_problem()
        for problem in cls.SUB_PROBLEMS:
            problem.init()

        return

    @classmethod
    def divide_problem(cls):
        undivided = [n for n in range(cls.ALL_NUM)]
        while len(undivided):
            _site_id = undivided[0]
            _site = cls.ALL_SITES[_site_id]
            for problem in cls.SUB_PROBLEMS:
                if problem.has_site(_site):
                    break
                elif problem.can_put(_site):
                    problem.put_site(_site)
                    undivided.remove(_site_id)
                    for _near_site_id in _site.dis0_50:
                        if _near_site_id not in undivided:
                            continue
                        elif problem.has_site(cls.ALL_SITES[_near_site_id]):
                            continue
                        else:
                            problem.put_site(cls.ALL_SITES[_near_site_id])
                            undivided.remove(_near_site_id)
                    else:
                        break  # 不一定要写else，为了看着方便还是写了
                else:
                    continue
            else:  # for被break中断则不执行
                problem = SubProblem()
                problem.id = len(cls.SUB_PROBLEMS)
                problem.put_site(_site)
                undivided.remove(_site_id)
                cls.SUB_PROBLEMS.append(problem)

        while True:
            _problem_num = len(cls.SUB_PROBLEMS)
            _change = False
            for i in range(1, _problem_num):
                for _site in cls.SUB_PROBLEMS[i].SITES:
                    for j in range(_problem_num):
                        if i == j:
                            continue
                        if cls.SUB_PROBLEMS[j].can_put(_site):  # 则这两个问题可以合并
                            for s in cls.SUB_PROBLEMS[i].SITES:
                                cls.SUB_PROBLEMS[j].put_site(s)
                            cls.SUB_PROBLEMS.pop(i)
                            _change = True
                            break
                    if _change:
                        break
                if _change:
                    break
            if _problem_num == len(cls.SUB_PROBLEMS):
                break

        for i in range(len(cls.SUB_PROBLEMS)):
            cls.SUB_PROBLEMS[i].id = i

        return

    @classmethod
    def summary_results(cls, gas):
        cls.CHOOSE = np.zeros(cls.ALL_NUM, int)
        cls.CONNECT = np.zeros([cls.ALL_NUM, cls.ALL_NUM], int)
        result = []
        _total_loss = 0
        _total_len = 0
        _host_num = 0
        _zone_num = 0
        _un_cover = 0
        _sate_num = 0
        total_cost = 0

        for _ga in gas:
            _re = dict()
            _sol = _ga.population[0].solution
            _re["iteration_num"] = _ga.current_epoch
            _re["best_fitness"] = float(_ga.population[0].fitness)
            _re["avg_loss"] = float(_sol.avg_loss)
            _sites = []
            for _s in _ga.problem.SITES:
                _sites.append({"id": _s.id, "no": _s.no, "kind": _s.kind, "lat": _s.lat, "lng": _s.lng})
            _re["sites"] = _sites
            _re["hosts"] = list(map(lambda x: int(x), _sol.hosts))
            _sol.create_tree()
            _re["plan"] = _sol.plan
            _re["zones"] = _sol.zones
            result.append(_re)
            for _z in _sol.zones:
                cls._prim(list(map(lambda x: _ga.problem.SITES[x].id, _z)))
            for _h, _plan in _sol.plan.items():
                cls.CHOOSE[int(_ga.problem.SITES[_h].id)] = 1
                _tree = _plan["tree"]
                for _edge in _tree:
                    for i in range(1, len(_edge)):
                        cls.CONNECT[int(_ga.problem.SITES[_edge[i]].id), int(_ga.problem.SITES[_edge[i - 1]].id)] = 1
            _total_loss += _re["avg_loss"] * len(_sol.children)
            _total_len += len(_sol.children)
            _host_num += len(_re["hosts"])
            _zone_num += len(_re["zones"])
            _sate_num += _sol.sate_num
            _un_cover += _sol.un_cover
            total_cost += _sol.all_cost

        path_loss = _total_loss / _total_len
        total_host_num = len(np.where(cls.CHOOSE == 1)[0])
        # total_cost = cls.ALL_NUM * 5 + _host_num * 5 + _sate_num * 50
        return total_cost, path_loss, result, gas, total_host_num, _un_cover, _sate_num

    @classmethod
    def _prim(cls, zone):  # todo
        vertex_num = len(zone)
        graph = []
        for host in zone:
            graph.append(list(map(lambda x: INF if cls.PRE_DIS[host, x] > 50 else cls.PRE_DIS[host, x], zone)))
        visit = [False] * vertex_num
        dis = [INF] * vertex_num
        pre_index = [0] * vertex_num
        for i in range(vertex_num):
            min_dist = INF + 1
            next_index = -1
            for j in range(vertex_num):
                if dis[j] < min_dist and not visit[j]:
                    min_dist = dis[j]
                    next_index = j
            # print(nextIndex)
            visit[next_index] = True
            for j in range(vertex_num):
                if dis[j] > graph[next_index][j] and not visit[j]:
                    dis[j] = graph[next_index][j]
                    pre_index[j] = next_index
        for i in range(vertex_num):
            if i != pre_index[i]:
                cls.CONNECT[zone[i], zone[pre_index[i]]] = 2
                cls.CONNECT[zone[pre_index[i]], zone[i]] = 2
        return

    @classmethod
    def save_file(cls):  # todo
        posi = []
        for i in range(cls.ALL_NUM):
            posi.append([cls.ALL_SITES[i].name, cls.CHOOSE[i]])
        p = save_csv(posi, "Posi")
        graph = []
        names = [" ", ]
        for i in range(cls.ALL_NUM):
            row = cls.CONNECT[i].tolist()
            row.insert(0, cls.ALL_SITES[i].name)
            graph.append(row)
            names.append(cls.ALL_SITES[i].name)
        graph.insert(0, names)
        g = save_csv(graph, "Graph")
        return p + "  " + g

    # 计算distance(i,j)矩阵
    @staticmethod
    def distance(num, lng1, lat1, lng2, lat2):
        lat1 = (math.pi / 180.0) * lat1
        lat2 = (math.pi / 180.0) * lat2
        lng1 = (math.pi / 180.0) * lng1
        lng2 = (math.pi / 180.0) * lng2
        temp = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lng2 - lng1)
        shit = np.where(temp > 1.0)  # 计算结果异常的点，由于精度损失可能会大于1
        temp[shit] = 1.0
        dis = np.arccos(temp) * 6378.0
        dis[(np.arange(0, num), np.arange(0, num))] = 0.0  # 由于精度损失，要设置对角线上全部为0
        return dis

    @staticmethod
    def create_chaotic_sequence(size):
        _chaotic_sequence = []
        _chaotic_number = random.uniform(0, 1)
        y = _chaotic_number
        for _c in range(100):
            x = y
            _chaotic_sequence.append(int((x * size + 1)))
            y = 4 * x * (1 - x)
        return _chaotic_sequence

    @classmethod
    def ex_summary_results(cls, gas, size, ex_time):
        ex_result = []
        cls.CHOOSE = np.zeros(cls.ALL_NUM, int)
        cls.CONNECT = np.zeros([cls.ALL_NUM, cls.ALL_NUM], int)
        _total_loss = 0
        _total_len = 0
        _host_num = 0
        _zone_num = 0
        _un_cover = 0
        _sate_num = 0
        total_cost = 0

        for _ga in gas:
            _sol = _ga.population[0].solution
            for _z in _sol.zones:
                cls._prim(list(map(lambda x: _ga.problem.SITES[x].id, _z)))
            for _h, _plan in _sol.plan.items():
                cls.CHOOSE[int(_ga.problem.SITES[_h].id)] = 1
                _tree = _plan["tree"]
                for _edge in _tree:
                    for i in range(1, len(_edge)):
                        cls.CONNECT[int(_ga.problem.SITES[_edge[i]].id), int(_ga.problem.SITES[_edge[i - 1]].id)] = 1
            _total_loss += float(_sol.avg_loss) * len(_sol.children)
            _total_len += len(_sol.children)
            _host_num += len(_sol.hosts)
            _zone_num += len(_sol.zones)
            _sate_num += _sol.sate_num
            _un_cover += _sol.un_cover
            total_cost += _sol.all_cost

        path_loss = _total_loss / _total_len
        total_host_num = len(np.where(cls.CHOOSE == 1)[0])
        # total_cost = cls.ALL_NUM * 5 + _host_num * 5 + _sate_num * 50
        name = cls.save_file()
        ex_result.append([ex_time, total_cost, path_loss, total_host_num, _un_cover, _sate_num, name])
        print("钱 ", total_cost, "， 损耗 ", path_loss, "， 宿主数 ", total_host_num, "， 未覆盖 ",
              _un_cover, "， 卫星 ", _sate_num, "文件 ", name)

        # 然后计算剩余的
        for i in range(1, size):
            _total_loss = 0
            _total_len = 0
            _host_num = 0
            _zone_num = 0
            _un_cover = 0
            _sate_num = 0
            total_cost = 0
            cls.CHOOSE = np.zeros(cls.ALL_NUM, int)
            cls.CONNECT = np.zeros([cls.ALL_NUM, cls.ALL_NUM], int)
            for _ga in gas:
                _sol = _ga.population[i].solution
                # for _z in _sol.zones:
                #     cls._prim(list(map(lambda x: _ga.problem.SITES[x].id, _z)))
                for _h, _plan in _sol.plan.items():
                    cls.CHOOSE[int(_ga.problem.SITES[_h].id)] = 1
                    _tree = _plan["tree"]
                    for _edge in _tree:
                        for i in range(1, len(_edge)):
                            cls.CONNECT[
                                int(_ga.problem.SITES[_edge[i]].id), int(_ga.problem.SITES[_edge[i - 1]].id)] = 1
                _total_loss += float(_sol.avg_loss) * len(_sol.children)
                _total_len += len(_sol.children)
                _host_num += len(_sol.hosts)
                _zone_num += len(_sol.zones)
                _sate_num += _sol.sate_num
                _un_cover += _sol.un_cover
                total_cost += _sol.all_cost
            path_loss = _total_loss / _total_len
            total_host_num = len(np.where(cls.CHOOSE == 1)[0])
            ex_result.append([ex_time, total_cost, path_loss, total_host_num, _un_cover, _sate_num, " "])

        return ex_result


class Individual(object):  # todo 基因与预设之间的关系
    def __init__(self, solution):
        self.solution = solution
        self.mutation_rate = 0.3  # 变异概率
        self.fitness = 0
        self.gene = self.solution.choose.tolist()

    def compute_fitness(self, a=1, b=0):
        eva_para = self.solution.evaluate()
        self.fitness = a * eva_para[0] + b * eva_para[1]
        return

    def update_itself(self, problem):
        _new_hosts = [i for i in range(self.solution.NUM) if self.gene[i] == 1]
        self.solution = problem.get_preset_solution(_new_hosts)
        self.compute_fitness()
        for i in range(self.solution.NUM):
            self.gene[i] = self.solution.choose[i]
        return

    def will_change(self):
        return True if random.uniform(0, 1) <= self.mutation_rate else False


class GeneticAlgorithm(object):
    def __init__(self, problem):
        self.problem = problem
        self.population_size = 0
        self.population = []
        self.previous_pop = []
        self.current_epoch = 0
        self.Chaotic_sequence = []

    # @func_timer
    def generating_initial_population(self, size):
        self.population_size = size
        self.create_chaotic_sequence()
        _population = []
        for i in range(size):
            _sol = self.problem.get_preset_solution()
            _individual = Individual(_sol)
            _individual.compute_fitness()
            _population.append(_individual)
        self.population = sorted(_population, key=lambda x: x.fitness)
        return

    def create_chaotic_sequence(self):
        _chaotic_sequence = []
        _chaotic_number = my_random()
        y = _chaotic_number
        for _c in range(100):
            x = y
            _chaotic_sequence.append(int((x * self.problem.NUM)))
            y = 4 * x * (1 - x)
        self.Chaotic_sequence = _chaotic_sequence
        return

    def chaotic_seq(self, step):
        _chaotic_sequence = []
        _chaotic_number = my_random()
        y = _chaotic_number
        for _c in range(step):
            x = y
            _chaotic_sequence.append(int((x * self.problem.NUM)))
            y = 4 * x * (1 - x)
        return _chaotic_sequence

    def _cross(self, sim=0.6):
        _pop = self.population
        seq = self.chaotic_seq(self.population_size * 5)
        for i in range(0, self.population_size, 2):
            _cos_sim = get_cosine_similarity(_pop[i].solution.choose, _pop[i + 1].solution.choose)  # 余弦相似度
            if _cos_sim < sim:  # todo 阈值调整
                # print(_pop[i].gene)
                a = seq[i]
                b = seq[i + 1]
                if a > b:
                    a, b = b, a
                for pos in range(a, b, 1):
                    _pop[i].gene[pos], _pop[i + 1].gene[pos] = _pop[i + 1].gene[pos], _pop[i].gene[pos]
                # print(self.population[i].gene)
                # print(0)

        return

    def _mutation(self):
        for ind in self.population:
            if True:  # ind.will_change():
                _sol = ind.solution
                for host in ind.solution.hosts:
                    _num = 4 if _sol.SITES[host].kind == 1 else 2
                    if _sol.plan[host]["sonNum"] < _num:
                        near_host = sorted(set(_sol.SITES[host].dis0_50) & set(_sol.hosts),
                                           key=lambda x: _sol.DIS[x, host])
                        if len(near_host) > 0:
                            ind.gene[host] = 0
                            ind.gene[near_host[0]] = 0
        return

    def _grow(self):
        for ind in self.population:
            ind.update_itself(self.problem)
        self.population = sorted(self.population, key=lambda x: x.fitness)
        return

    def _select(self):
        elites_num = int(self.population_size * 0.3)
        _temp_pop = self.previous_pop[:elites_num]
        _temp_pop += self.population[:self.population_size - elites_num]
        self.population = sorted(_temp_pop, key=lambda x: x.fitness)
        return

    def avg_fit(self):
        fit = sum(map(lambda x: x.fitness, self.population)) / self.population_size
        return fit

    def run_one_time(self, sim):
        self.previous_pop = copy.deepcopy(self.population)
        self._cross(sim)
        self._mutation()
        self._grow()
        self._select()  # 精英策略
        self.current_epoch += 1
        return


def save_csv(mat, name):
    file_name = time.strftime("%m%d%H%M%S_", time.localtime()) + name + ".csv"
    path = "result\\" + file_name
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(mat)):
            writer.writerow(mat[i])
        print("------csv文件已保存: ", path)
    return file_name


def save_json(result):
    file_name = time.strftime("%m%d%H%M%S_info", time.localtime()) + ".json"
    path = "result\\" + file_name
    with open(path, "w") as f:
        json.dump(result, f)
        print("------方案数据已保存: ", path)
    return file_name


def solve_problem(problem, population_size, iteration_num):
    print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
          " 开始" + "problem " + str(problem.id) + "]")
    t0 = time.time()

    ga = GeneticAlgorithm(problem)
    ga.generating_initial_population(population_size)
    init_fit = ga.avg_fit()
    last_fitness = init_fit
    sim = 0.6
    while ga.current_epoch < iteration_num:
        ga.run_one_time(sim)
        sim = 0.6
        log(ga.current_epoch)
        now_fit = ga.avg_fit()
        print(now_fit)
        if now_fit > last_fitness:
            last_fitness = now_fit
        elif (now_fit / last_fitness) > 0.999:
            if now_fit > init_fit:
                last_fitness = now_fit
                sim = 0.2
                continue
            else:
                break
        else:
            last_fitness = now_fit

    host_num = len(ga.population[0].solution.hosts)

    ho_list = []
    for ho in ga.population[0].solution.hosts:
        num = 4 if ga.population[0].solution.SITES[ho].kind == 1 else 2
        if ga.population[0].solution.plan[ho]["sonNum"] < num:
            ho_list.append(ho)
    print("num:", len(ho_list), " :", ho_list)

    path_adjust_num = 1
    for i in range(int(host_num / 2)):
        ch = ga.population[0].solution.adjust_between_group()
        ga.population[0].solution.create_tree()
        if ch:
            ch = ga.population[0].solution.adjust_in_group()
            ga.population[0].solution.create_tree()
            path_adjust_num += 1
            if not ch:
                break

    t1 = time.time()
    log("adjust_num: " + str(path_adjust_num))
    print("[当前时间:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
          "problem " + str(problem.id) + "耗时: {time:.2f}s]".format(time=t1 - t0))

    return ga


@func_timer
def algorithm(data, population_size=20, iteration_num=10):
    Pre.process_data(data)
    ga_list = list()
    # pool = multiprocessing.Pool(processes=2)
    for sub_problem in Pre.SUB_PROBLEMS:
        ga_list.append(solve_problem(sub_problem, population_size, iteration_num))
        # pool.apply_async(solve_problem, args=(sub_problem, population_size, iteration_num, queue))

    return Pre.summary_results(ga_list)


def read_data(size):
    data = list()
    wb = xlrd.open_workbook("B题测试数据-更新.xlsx")
    sheet = wb.sheets()[0]

    for i in range(size):
        row = i + 2
        name = str(sheet.cell_value(row, 0))
        lng = float(sheet.cell_value(row, 1))
        lat = float(sheet.cell_value(row, 2))
        kind_name = str(sheet.cell_value(row, 3))
        kind = 1 if kind_name == "Butterfly Site" else 0
        data.append({"lat": lat, "lng": lng, "site_name": name,
                     "kind": kind, "preset": False})  # random.randint(0, 1)

    return data


def main():
    _data = read_data(1000)  # get_random_data(2229)  # 112.9, 114.4, 22.5, 23.4

    _re = algorithm(_data, POPULATION_SIZE, ITERATION_NUM)

    log("total_cost: " + str(_re[0]) + ", path_loss: " + str(_re[1]))
    print("host_num: ", _re[4], "  uncover num: ", _re[5], "  sate num: ", _re[6])

    total_result = dict()
    total_result["total_cost"] = _re[0]
    total_result["path_loss"] = _re[1]
    total_result["host_num"] = _re[4]
    total_result["un_cover_num"] = _re[5]
    total_result["sate_num"] = _re[6]
    total_result["problems"] = _re[2]
    Pre.save_file()
    save_json(total_result)

    return


def one_experiment(data, i, size):
    Pre.process_data(data)
    ga_list = []
    for problem in Pre.SUB_PROBLEMS:
        ga = GeneticAlgorithm(problem)
        ga.generating_initial_population(size)
        ga_list.append(ga)
    res = Pre.ex_summary_results(ga_list, size, i + 1)
    del ga_list
    return res


def experiment(data):
    res = list()
    res.append(["第x次生成实验", "总体成本", "平均损耗", "宿主站数", "未覆盖数", "卫星数", "文件名"])

    for i in range(20):
        res += one_experiment(data, i, 30)
    save_csv(res, "实验结果汇总")


if __name__ == "__main__":
    #  遗传算法参数
    POPULATION_SIZE = 50  # 种群规模
    ITERATION_NUM = 50  # 迭代次数

    main()
    # experiment(read_data(500))

'''
TODO 多进程并行计算

个体更新时不用计算回传损耗？

1. 子站与基站最远20，基站与基站最远为50，同一小组子站与基站间接最远为40=20+10+10（跳数≤3）
预设全部为host，一步一步调整。  试试DQN
'''
