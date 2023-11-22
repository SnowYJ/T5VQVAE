from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def bleurt_sim(gold, pred):
    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    model.eval()
    references = [gold]
    candidates = [pred]
    with torch.no_grad():
        scores = model(**tokenizer(references, candidates, return_tensors='pt'))[0].squeeze()

    print(scores)


if __name__ == '__main__':
    pass