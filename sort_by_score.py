import argparse
from tqdm import tqdm
from operator import itemgetter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-res_dir', default='', type=str)
parser.add_argument('-epoch', default='', type=str)
args = parser.parse_args()

def sort(test, scores, sorted_test_path):
    print('Sorting tested file to: '+sorted_test_path)
    new_test = []
    for line in test:
        if(line[0] == '#'):
            new_test.append(line)
        else:
            new_test.append(' '.join(line.split(' ')[:2])+'\n')
    test = new_test
    scores = [float(line.strip('\n').replace(' ', '\t').split('\t')[-1]) for line in scores]

    data = dict()
    key = test[0]
    value = []

    for line in tqdm(test[1:],desc='Loading original test file...'):
        if(line[0] == '#'):
            data[key] = value
            key = line
            value = []
        else:
            value.append(line)

    count = 0
    
    with open(sorted_test_path,'w',encoding='utf-8') as sorted_test:
        for user, tags in tqdm(data.items(),desc='Writing sorted test file...'):
            temp_scores = scores[count:count+len(tags)]
            sort_tags, sort_scores = [list(x) for x in zip(*sorted(zip(tags, temp_scores), key=itemgetter(1), reverse=True))]
            sorted_test.write(user)
            sorted_test.writelines(sort_tags)
            count += len(tags)

def main():
    test = open(f'{args.res_dir}/test.dat','r',encoding='utf-8').readlines()
    score = open(f'{args.res_dir}/test_{args.epoch}_score.txt','r',encoding='utf-8').readlines()
    sorted_test = f'{args.res_dir}/sorted_test_{args.epoch}.dat'
    sort(test, score, sorted_test)
    
if __name__ == '__main__':
   main()
