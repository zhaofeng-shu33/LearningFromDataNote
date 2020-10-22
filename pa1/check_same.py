import os
import csv
if __name__ == "__main__":
    # load github_id to student_name
    dic = {}
    with open('classroom_roster.csv', newline='') as cf:
        r = csv.reader(cf, delimiter=',')
        for row in r:
            _id = row[1].strip('"')
            name = row[0].strip('"')
            dic[_id] = name
    print(dic)
    Ls = []
    f = open('build/differences.txt')
    st = f.read()
    for k, v in dic.items():
        if k == '':
            continue
        st = st.replace(k, k + '___' + v)
    with open('build/differences-edit.txt', 'w') as f2:
        f2.write(st)
