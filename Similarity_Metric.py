from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW, Adam, Adamax
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import pandas as pd
from multiprocessing import cpu_count
import time

Device = torch.device("cpu")
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class STSBDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        similarity_scores = [i['label'] for i in dataset]
        self.normalized_similarity_scores = [i/5.0 for i in similarity_scores]
        self.first_sentences = [i['sentence1'] for i in dataset]
        self.second_sentences = [i['sentence2'] for i in dataset]
        self.concatenated_sentences = [[str(x), str(y)] for x,y in   zip(self.first_sentences, self.second_sentences)]

    def __len__(self):
        return len(self.concatenated_sentences)

    def get_batch_labels(self, idx):
        return torch.tensor(self.normalized_similarity_scores[idx])

    def get_batch_texts(self, idx):
        return Tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

def collate_fn(Texts):
    Input_Ids = Texts['input_ids']
    Attention_Masks = Texts['attention_mask']
    Features = [{'input_ids': Input_Id, 'attention_mask': Attention_Mask}
                for Input_Id, Attention_Mask in zip(Input_Ids, Attention_Masks)]
    return Features


class Bert_For_STS(torch.nn.Module):

    def __init__(self):
        super(Bert_For_STS, self).__init__()
        self.Bert = models.Transformer('bert-base-uncased', max_seq_length=128)
        self.Pooling_Layer = models.Pooling(self.Bert.get_word_embedding_dimension())
        self.Sts_Bert = SentenceTransformer(modules=[self.Bert, self.Pooling_Layer])

    def forward(self, Inp):
        Outp = self.Sts_Bert(Inp)['sentence_embedding']
        return Outp


class Cosine_Similarity_Loss(torch.nn.Module):

    def __init__(self,  Loss_Fn=torch.nn.MSELoss(), Transform_Fn=torch.nn.Identity()):
        super(Cosine_Similarity_Loss, self).__init__()
        self.Loss_Fn = Loss_Fn
        self.Transform_Fn = Transform_Fn
        self.Cos_Similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, Inps, Labels):
        Emb_1 = torch.stack([inp[0] for inp in Inps])
        Emb_2 = torch.stack([inp[1] for inp in Inps])
        Outps = self.Transform_Fn(self.Cos_Similarity(Emb_1, Emb_2))
        return self.Loss_Fn(Outps, Labels.squeeze())


def Train_Model(Epochs, Batch_Size, Optimizer_Name, Learning_Rate = 0.000001, Save_Path = None, Show_Stat = False):
    New_Model = Bert_For_STS()
    New_Model = New_Model.to(Device)
    Optimizer = None
    match Optimizer_Name:
        case "Adam":
            Optimizer = Adam(New_Model.parameters(), lr=Learning_Rate)
        case "AdamW":
            Optimizer = AdamW(New_Model.parameters(), lr=Learning_Rate)
        case "Adamax":
            Optimizer = Adamax(New_Model.parameters(), lr=Learning_Rate)
    if not(Optimizer == None):
        Dataset = load_dataset("glue", "stsb")
        Train_Ds = STSBDataset(Dataset['train'])
        Val_Ds = STSBDataset(Dataset['validation'])
        Train_D_Loader = DataLoader(
            Train_Ds,
            num_workers=cpu_count(),
            batch_size=Batch_Size,
            shuffle=True
        )
        Val_D_Loader = DataLoader(
            Val_Ds,
            num_workers=cpu_count(),
            batch_size=Batch_Size
        )
        T_Steps = len(Train_D_Loader) * Epochs
        Scheduler = get_linear_schedule_with_warmup(Optimizer, num_warmup_steps=0, num_training_steps=T_Steps)
        Criterion = Cosine_Similarity_Loss()
        Seed_Val = 42
        random.seed(Seed_Val)
        torch.manual_seed(Seed_Val)
        Train_Stats = []
        print("Training Process Started")
        for i in range(0, Epochs):
            T_T_Loss = 0
            print("Epochs " + str(i + 1) + " Training Started")
            New_Model.train()
            for T_Data, T_Label in tqdm(Train_D_Loader):
                T_Data['input_ids'] = T_Data['input_ids'].to(Device)
                T_Data['attention_mask'] = T_Data['attention_mask'].to(Device)
                T_Data = collate_fn(T_Data)
                New_Model.zero_grad()
                Outp = [New_Model(Feature) for Feature in T_Data]
                Loss = Criterion(Outp, T_Label.to(Device))
                T_T_Loss = T_T_Loss + Loss.item()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_(New_Model.parameters(), 1.0)
                Optimizer.step()
                Scheduler.step()
            Avg_T_Loss = T_T_Loss / len(Train_D_Loader)
            print("Training Finished, Avg_Loss: " + str(Avg_T_Loss))
            New_Model.eval()
            T_Eval_Loss = 0
            print("Epochs " + str(i + 1) + " Testing Process Started")
            for V_Data, V_Label in tqdm(Val_D_Loader):
                V_Data['input_ids'] = V_Data['input_ids'].to(Device)
                V_Data['attention_mask'] = V_Data['attention_mask'].to(Device)
                V_Data = collate_fn(V_Data)
                with torch.no_grad():
                    Outp = [New_Model(Feature) for Feature in V_Data]
                Loss = Criterion(Outp, V_Label).to(Device)
                T_Eval_Loss = T_Eval_Loss + Loss.item()
            Avg_V_Loss = T_Eval_Loss / len(Val_D_Loader)
            print("Testing Finished, Avg_Loss: " + str(Avg_V_Loss))
            Train_Stats.append(
                {
                    'Epoch': i + 1,
                    'Training Loss': Avg_T_Loss,
                    'Testing Loss': Avg_V_Loss
                }
            )
        if Show_Stat:
            See_Stat = pd.DataFrame(data=Train_Stats)
            See_Stat = See_Stat.set_index('Epoch')
            print(See_Stat)
        if Save_Path == None:
            return New_Model, Train_Stats
        else:
            torch.save(New_Model, Save_Path + "Model(Similarity)_" + str(time.time()).replace(".", "") + ".pth")
            print("Model Saved At: " + Save_Path)
            return True
    print("Wrong Optimizer Name, Impossible To Train...")
    return False


def Predict(Sentence_1, Sentence_2, Model):
    Model.eval()
    Inp = Tokenizer([Sentence_1, Sentence_2], padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    del Inp["token_type_ids"]
    Outp = Model(Inp)
    Res = torch.nn.functional.cosine_similarity(Outp[0], Outp[1], dim=0).item()
    return Res


def Predict_Arr(Main_Sentence, Sentences_Arr, Model):
    Similarity_Each = []
    for Sentence in tqdm(Sentences_Arr):
        Similarity_Each.append(Predict(Main_Sentence, Sentence, Model))
    Avg_Similarity = sum(Similarity_Each) / len(Similarity_Each)
    return Similarity_Each, Avg_Similarity