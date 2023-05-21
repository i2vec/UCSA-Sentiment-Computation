import json
from datasets import load_dataset


def get_dataset(balance=False):
    datafiles = {
        'train': '/home/xumingjun/from17/sentiment-analysis/02-bert-textcnn/data/train.tsv', 
        'dev': '/home/xumingjun/from17/sentiment-analysis/02-bert-textcnn/data/dev.tsv', 
        'test': '/home/xumingjun/from17/sentiment-analysis/02-bert-textcnn/data/test.tsv'
    }
    go_emotions = load_dataset('csv', data_files=datafiles, delimiter='\t')
    
    base2id, id2base, emotion2id, id2emotion = {}, {}, {}, {}
    
    with open('/home/xumingjun/from17/sentiment-analysis/02-bert-textcnn/data/emotions_base.txt', 'r') as f:
        emotions_base = f.readlines()

    for base in emotions_base:
        id = len(base2id)
        base = base.strip('\n').strip(' ')
        base2id[base] = id
        id2base[id] = base
        
    with open('/home/xumingjun/from17/sentiment-analysis/02-bert-textcnn/data/emotions.txt', 'r') as f:
        emotions = f.readlines()

    for emotion in emotions:
        id = len(emotion2id)
        emotion = emotion.strip('\n').strip(' ')
        emotion2id[emotion] = id
        id2emotion[id] = emotion
        
    with open('/home/xumingjun/from17/sentiment-analysis/02-bert-textcnn/data/label_groups.json', 'r') as f:  
        groups = json.load(f)
        
    emotionid2baseid = {}
    for k, vs in groups.items():
        baseid = base2id[k]
        for v in vs:
            emotionid = emotion2id[v]
            emotionid2baseid[emotionid] = baseid
            
    def map_label(example):
        example['label'] = emotionid2baseid[example['label']]
        return example

    go_emotions = go_emotions.map(map_label)
    
    return go_emotions