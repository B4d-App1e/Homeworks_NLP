import Similarity_Metric as SM
import torch
from transformers import pipeline, set_seed, GPT2Tokenizer
from transformers import logging as Logs
from tqdm import tqdm
import random
import json
import time
import copy


def Save_As_Json(File_Path, File_Name, Initial_Sent, Sents_Arr, Similars_Arr):
    with open(File_Path + File_Name, "w+") as File:
        File.write(json.dumps({'Sentence': Initial_Sent, 'Similar_Sentences': Sents_Arr, 'Similarity_Scores': Similars_Arr}))
        File.close()


if __name__ == '__main__':
    Logs.set_verbosity_error()
    Save_Model_Path = "/Your_Path_For_Model/"
    Model_Name = "Model(Similarity)_1703909050553345.pth"
    Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    Mask_Replacer = pipeline('fill-mask', model='bert-base-uncased')
    Similarity_Tester = torch.load(Save_Model_Path + Model_Name)
    #Word_Generator = pipeline(
    #    "text-generation",
    #    model="tiiuae/falcon-7b-instruct",
    #    tokenizer=Tokenizer,
    #    torch_dtype=torch.bfloat16,
    #    device_map='auto'
    #)
    Word_Generator_try_2 = pipeline('text-generation', model='gpt2')
    Initial_Sentence = input("Enter Sentence for Which You Want To Create Similar Ones: ")
    Num_Words_Mask = 0
    Similarity_Limit = 0
    Init_Sent_Arr = Initial_Sentence.split()
    Models_Pick = -1
    while True:
        Num_Buf = input("What Model You Wanna Use?( 0 - Masking, 1 - Generative, 2 - Both): ")
        try:
            Models_Pick = int(Num_Buf)
            if -1 < Models_Pick < 3:
                break
            else:
                print("Entered Number Is Out Of Range, Try Another One...")
        except ValueError:
            print("Only Integer Values Required, Try Again...")
            pass
    if Models_Pick == 0 or Models_Pick == 2:
        while True:
            Num_Buf = input("Enter Number Of Words To Be Masked: ")
            try:
                Num_Words_Mask = int(Num_Buf)
                if -1 < Num_Words_Mask <= len(Init_Sent_Arr):
                    break
                elif Num_Words_Mask > len(Init_Sent_Arr):
                    print("Number Of Masked Words Can't Be Greater Than Number of Words Itself, Try Again...")
                else:
                    print("Number Should Be Positive, Try Again...")
            except ValueError:
                print("Only Integer Values Required, Try Again...")
                pass
    while True:
        Num_Buf = input("Enter Limit Of Similarity For Generated Sentences(from 0 to 100): ")
        try:
            Similarity_Limit = int(Num_Buf) / 100
            if 0.0 <= Similarity_Limit <= 1.0:
                break
            elif Similarity_Limit > 1.0:
                print("Similarity Limit Can't be Grater Than 100, Try Again...")
            else:
                print("Similarity Limit Should be Positive, Try Again...")
        except ValueError:
            print("Only Integer Values Required, Try Again...")
            pass
    if len(Init_Sent_Arr) > 1 and len(Init_Sent_Arr) < 101:
        Unmasked_Formed_Sents = [Initial_Sentence]
        if Models_Pick == 0 or Models_Pick == 2:
            #Num_Words_Mask = random.randint(1, round(0.5 * len(Init_Sent_Arr)))
            All_Ids = list(range(len(Init_Sent_Arr)))
            Mask_Ids = []
            for i in range(0, Num_Words_Mask):
                El_Add = random.randint(0, len(All_Ids) - 1)
                Mask_Ids.append(All_Ids[El_Add])
                del All_Ids[El_Add]
            #Ids To be Masked Formed Here
            print("Creating Basic Variants Via Masking...")
            for Val in tqdm(Mask_Ids):
                Sents_To_Work_With = copy.deepcopy(len(Unmasked_Formed_Sents))
                for i in range(Sents_To_Work_With):
                    Sent_Arr = Unmasked_Formed_Sents[i].split()
                    Masked_Sent_Arr = Sent_Arr
                    Masked_Sent_Arr[Val] = "[MASK]"
                    Masked_Sent = ' '.join(Masked_Sent_Arr)
                    Unmasked_Sents = Mask_Replacer(Masked_Sent)
                    if len(Unmasked_Sents) > 5:
                        Unmasked_Sents = Unmasked_Sents[:5]
                    for Sent in Unmasked_Sents:
                        Sent_j = json.loads(json.dumps(Sent))
                        Buf_Sent = Sent_j["sequence"]
                        Buf_Sent.replace("[CLS] ", "")
                        Buf_Sent.replace(" [SEP]", "")
                        Unmasked_Formed_Sents.append(Buf_Sent)
            #Unmasked Sentences Created Here
            Unmasked_Formed_Sents = list(map(lambda x: x.lower(), Unmasked_Formed_Sents))
            Unmasked_Formed_Sents = list(dict.fromkeys(Unmasked_Formed_Sents))
        Augmented_Formed_Sents = []
        if Models_Pick == 1 or Models_Pick == 2:
            print("Creating Variants By Generating End Of Sentence Based On Its Start...")
            for Sentence in tqdm(Unmasked_Formed_Sents):
                Seed = random.randint(0, 42)
                set_seed(Seed)
                Sent_Arr = Sentence.split()
                Num_Words_Del = random.randint(1, round(0.5 * len(Sent_Arr)))
                Short_Sent_Arr = copy.deepcopy(Sent_Arr)
                del Short_Sent_Arr[-Num_Words_Del:]
                Short_Sent = ' '.join(Short_Sent_Arr)
                Length_Lim = random.randint(len(Sent_Arr), round(1.25 * len(Sent_Arr)))
            #    Augmented_Sents = Word_Generator(
            #        Short_Sent,
            #        max_length=Length_Lim,
            #        do_sample=True,
            #        top_k=5,
            #        num_return_sequences=5,
            #        eos_token_id=Tokenizer.eos_token_id
            #    )
                Augmented_Sents = Word_Generator_try_2(Short_Sent, max_length=Length_Lim, num_return_sequences=5, pad_token_id=Tokenizer.eos_token_id)
                for i in range(len(Augmented_Sents)):
                    Sent_j = json.loads(json.dumps(Augmented_Sents[i]))
                    Buf_Sent = Sent_j['generated_text']
                    Augmented_Formed_Sents.append(Buf_Sent)
            Augmented_Formed_Sents = list(map(lambda x: x.lower(), Augmented_Formed_Sents))
        #Sentences Should Be Both Unmasked And Augmented at This Point
        print("Calculating Similarities Scores...")
        Unmasked_Formed_Sents.pop(0)
        Similar_Sents = Unmasked_Formed_Sents + Augmented_Formed_Sents
        Similar_Sents = list(dict.fromkeys(Similar_Sents))
        Similarity_Each, _ = SM.Predict_Arr(Initial_Sentence, Similar_Sents, Similarity_Tester)
        Similarity_Each, Similar_Sents = (list(t) for t in zip(*sorted(zip(Similarity_Each, Similar_Sents), reverse=True)))
        if Similarity_Each[len(Similarity_Each) - 1] < Similarity_Limit:
            Idx = next(x for x, val in enumerate(Similarity_Each) if val <= Similarity_Limit)
            Similarity_Each = Similarity_Each[:Idx]
            Similar_Sents = Similar_Sents[:Idx]
        Avg_Similarity = sum(Similarity_Each) / len(Similarity_Each)
        print("Created Similar Sentences and Their Similarity Scores:")
        for i in range(0, len(Similarity_Each)):
            print(Similar_Sents[i] + " – " + str(Similarity_Each[i]))
        print("Average Similarity: " + str(Avg_Similarity))
        Res_Name = "Result_" + str(time.time()).replace(".", "") + ".json"
        Save_As_Json(Save_Model_Path, Res_Name, Initial_Sentence, Similar_Sents, Similarity_Each)
        print("Result Saved at: " + Save_Model_Path + Res_Name)
    elif len(Init_Sent_Arr) < 2:
        print("More Words Required!")
    else:
        print("Maximum Number Of Words In Sentence Is Limited to 100")
    #Res = SM.Train_Model(10, 15, "AdamW", Save_Path=Save_Model_Path, Show_Stat=True)
    #if Res:
    #    print("Train Process Done Successfully!")
    #else:
    #    print("Train Process Failed!")