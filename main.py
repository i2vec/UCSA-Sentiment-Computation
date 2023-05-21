import argparse
from models import BertTextCNN, BertForClassification
from data_prepare import get_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
from transformers import EarlyStoppingCallback
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from losses import FocalLoss, AsymmetricLoss, SmoothLabelCritierion
import pandas as pd


def main():
    # get model
    # alpha = torch.FloatTensor([0.048543689320388356, 0.38834951456310685, 0.38834951456310685, 0.012944983818770227, 0.09708737864077671, 0.048543689320388356, 0.016181229773462782])
    # model = BertTextCNN(args.num_classes, SmoothLabelCritierion(label_smoothing=0.1))
    # model = BertTextCNN(args.num_classes)
    model = BertTextCNN(args.num_classes, FocalLoss(alpha=torch.FloatTensor([1.2, 2, 2, 0.7, 1, 1, 0.8]), gamma=2))
    # model = BertForClassification(args.num_classes)
    # get bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # prepare data
    go_emotions = get_dataset()
    def data_process(data):
        return tokenizer(
            data['text'], 
            padding='max_length', 
            max_length=args.max_length, 
            truncation=True, 
            return_tensors='pt'
        )
    go_emotions = go_emotions.map(data_process, batched=True)
    go_emotions.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # training config
    training_args = TrainingArguments(
        output_dir=f'results/{args.name}', 
        learning_rate=args.lr, 
        logging_dir=f'logs/{args.name}', 
        logging_strategy="epoch", 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2, 
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size, 
        num_train_epochs=args.epochs, 
        load_best_model_at_end=True, 
        metric_for_best_model='accuracy'
    )
    
    # early stop
    earlystop_callback = EarlyStoppingCallback(early_stopping_patience=50, early_stopping_threshold=0.0005)
    # acc f1, precision, recall
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

    # datacollater
    def data_collator(data):
        input_ids = torch.cat([item['input_ids'].unsqueeze(0) for item in data], dim=0)
        attention_mask = torch.cat([item['attention_mask'].unsqueeze(0) for item in data], dim=0)
        labels = torch.tensor([item['label'] for item in data])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator, 
        train_dataset=go_emotions['train'], 
        eval_dataset=go_emotions['dev'], 
        compute_metrics=compute_metrics, 
        callbacks=[earlystop_callback]
    )
    

    
    if not args.eval:
        trainer.train()
    else:
        assert args.chpt is not None
        state_dict = torch.load(args.chpt, map_location=torch.device('cuda'))
        model.load_state_dict(state_dict)   
        test_res = trainer.evaluate(go_emotions['test'])
        res = trainer.predict(go_emotions['test'])
        import numpy as np
        pd.DataFrame(
            {
                "infer": list(np.argmax(res.predictions, axis=-1)), 
                "gt": res.label_ids
            }
        ).to_csv('./data/infer.tsv', sep='\t', index=False)
        print(test_res)
    
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--chpt', type=str, default=None)
    parser.add_argument('--name', type=str, default='unnamed')
    args = parser.parse_args()
    models = {
        'bert': 'bert-base-uncased', 
        'roberta': 'roberta-base', 
        'roberta-large': 'roberta-large'
    }
    assert args.model in models.keys()
    args.model = models[args.model]
    return args


if __name__ == "__main__":
    args = get_args()
    main()