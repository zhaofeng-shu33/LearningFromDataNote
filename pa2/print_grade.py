import os
import pickle
import csv
def load_dic():
    dic = {}
    with open('classroom_roster.csv', newline='') as cf:
        r = csv.reader(cf, delimiter=',')
        for row in r:
            _id = row[1].strip('"')
            name = row[0].strip('"')
            dic[name] = _id
    return dic
if __name__ == '__main__':
    f = open('build/student_list.txt')
    Ls = f.read().split('\n')
    Ls.pop(-1)
    f = open('build/grade.pickle', 'rb')
    Ls2 = pickle.load(f)
    dic = {}
    for i in Ls2:
        dic[i[0]] = i[1]
    dic2 = load_dic()
    for i in Ls:
        if dic.get(i) and dic[i] == '0':
            print(i, dic2[i], dic[i])

