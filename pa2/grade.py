import os
import subprocess
import csv
import pickle

prefix = './build/Programming Assignment 2 -- Neural Network-11-09-2020-12-02-42'

def load_dic():
    dic = {}
    with open('classroom_roster.csv', newline='') as cf:
        r = csv.reader(cf, delimiter=',')
        for row in r:
            _id = row[1].strip('"')
            name = row[0].strip('"')
            dic[_id] = name
    return dic
   
if __name__ == '__main__':
    dic = load_dic()
    Ls = []
    for i in os.listdir(prefix):
        if dic.get(i) is None:
            continue
        target_dir = os.path.join(prefix, i)
        sp = subprocess.run(["python3", "test.py"], cwd=target_dir, timeout = 5 * 60, capture_output=True)
        # timeout: 5 min
        output_str = sp.stdout.decode('utf-8')
        try:
            # extract score from output_str
            sentence_with_score = output_str.split('\n')[-2]
            score_str_val = sentence_with_score.split('PA2:')[-1].strip()
        except Exception as e:
            print(output_str)
        appended_value = [dic.get(i), score_str_val]
        print(appended_value)
        Ls.append(appended_value)
    with open('build/grade.pickle', 'wb') as f:
        pickle.dump(Ls, f)