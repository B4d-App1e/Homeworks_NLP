from gensim.parsing.preprocessing import remove_stopwords
import nltk
import pymorphy2
from nltk.corpus import stopwords
from dataclasses import dataclass
from typing import Iterator
import gzip
from tqdm import tqdm
import os
import time
import json
from multiprocessing import cpu_count
from gensim.models.word2vec import Word2Vec
from gensim.models import doc2vec
from sklearn.ensemble import RandomForestClassifier
import joblib

morph = pymorphy2.MorphAnalyzer()
nltk.download('stopwords')

def Read_File(Full_Path_W_Name, Res_As_List):
    Res = ""
    with open(Full_Path_W_Name, 'r') as File:
        Res = File.read()
        File.close()
    if Res_As_List:
        return Res.split("\n")
    else:
        return Res


def Clean_Text(Text, As_List = False, Res_Type = 0):
    #Text - string or list of strings, depends on As_List value,
    # As_List - bool,
    # Res_Type - int, 0 - String (Text), 1 - list of strings (Text as Lines), 2 - list of lists of strings (Text as Words)
    Txt_List = []
    Res = []
    if As_List:
        for Buf_Txt in Text:
            Buf = ""
            if Buf_Txt[len(Buf_Txt) - 1] == '.':
                Buf = Buf_Txt[:-1]
                Buf = Buf + ". "
            else:
                Buf = Buf_Txt
            Txt_List.append(Buf)
    else:
        Txt_Buf = Text.replace(".\n", ". \n")
        if Txt_Buf[len(Txt_Buf) - 1] == '.':
            Txt_Buf = Txt_Buf[:-1]
            Txt_Buf = Txt_Buf + ". "
        Txt_List = Txt_Buf.split("\n")
    Txt_List_Buf = ' '.join(Txt_List)
    Txt_List = Txt_List_Buf.split('. ')
    for Txt_Part in Txt_List:
        Txt_Part_Buf = Txt_Part.replace(",", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("(", "")
        Txt_Part_Buf = Txt_Part_Buf.replace(")", "")
        Txt_Part_Buf = Txt_Part_Buf.replace(":", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("\"", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("'", "")
        Txt_Part_Buf = Txt_Part_Buf.replace(" -", "")
        Txt_Part_Buf = Txt_Part_Buf.replace(" —", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("«", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("»", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("\\", "")
        Txt_Part_Buf = Txt_Part_Buf.replace("/", "")
        if Txt_Part_Buf == "":
            continue
        Txt_Buf = remove_stopwords(Txt_Part_Buf, stopwords.words()).split()
        Sub_Res = []
        for Word in Txt_Buf:
            W_mod = morph.parse(Word)[0]
            Sub_Res.append(W_mod.normal_form)
        if not(Res_Type == 2):
            Res.append(' '.join(Sub_Res))
        else:
            Res = Res + Sub_Res
    if not(Res_Type == 0):
        return Res
    else:
        return '\n'.join(Res)


def Create_Data_Set_W2V(Texts_Arr):
    Res = []
    print("Creating Word To Vec Dataset:")
    for Text in tqdm(Texts_Arr):
        Res.append(Clean_Text(Text.text, Res_Type=2))
    return Res


@dataclass
class Text:
    label: str
    title: str
    text: str


def Read_Texts(fn: str) -> Iterator[Text]:
    with gzip.open(fn, "rt", encoding="utf-8") as File:
        for Line in File:
            yield Text(*Line.strip().split("\t"))


def Save_Ds_To_File(Data_Set, File_Name, File_Path, Set_Classes = None):
    with open(File_Path + File_Name, "w+") as File:
        if Set_Classes == None:
            File.write(json.dumps({'DataSet': Data_Set}))
        else:
            File.write(json.dumps({'DataSet': Data_Set, "Answers": Set_Classes}))
        File.close()


def Read_Ds(Full_Path, W_Ans = False):
    Buf = None
    with open(Full_Path, "r") as File:
        Buf = json.load(File)
    if W_Ans:
        return Buf['DataSet'], Buf['Answers']
    return Buf['DataSet']


def Texts_To_Vec(Texts_Arr, Model, Neural_Outp = False, Doc_2_Vec = False):
    Res = []
    Sub_Res_Classes = []
    Unique_Classes = []
    print("Converting Text to Vectors")
    for Text in tqdm(Texts_Arr):
        Sub_Res = []
        Cl_Text = Clean_Text(Text.text, Res_Type=2)
        if Text.label in Unique_Classes:
            Sub_Res_Classes.append(Unique_Classes.index(Text.label))
        else:
            Sub_Res_Classes.append(len(Unique_Classes))
            Unique_Classes.append(Text.label)
        if Doc_2_Vec:
           Sub_Res = Model.infer_vector(Cl_Text).tolist()
        else:
            First = True
            for Word in Cl_Text:
                try:
                    if First:
                        Sub_Res = Model.wv[Word]
                        First = False
                    else:
                        Sub_Res = map(sum, zip(Sub_Res, Model.wv[Word]))
                except ValueError:
                    print("Text Contains Unknown Word: " + Word)
            Sub_Res = [x / len(Cl_Text) for x in Sub_Res]
        Res.append(Sub_Res)
    Res_Classes = []
    if Neural_Outp:
        for Val in Sub_Res_Classes:
            Val_Res = [0] * len(Unique_Classes)
            Val_Res[Val] = 1
            Res_Classes.append(Val_Res)
    else:
        Res_Classes = Sub_Res_Classes
    return Res, Res_Classes


def tagged_document(list_of_ListOfWords):
    for x, ListOfWords in enumerate(list_of_ListOfWords):
        yield doc2vec.TaggedDocument(ListOfWords, [x])


def Upload_Classes(Full_Path):
    Buf = None
    with open(Full_Path, "r") as File:
        Buf = File.read()
        File.close()
    Buf_Arr = Buf.split(", ")
    Res_Classes = []
    Res = []
    for Val in Buf_Arr:
        try:
            Buf_Res = Val.split(" - ")
            Res_Classes.append(int(Buf_Res[0]))
            Res.append(Buf_Res[1])
        except ValueError:
            print("File With Classes Has Wrong Structure, Try Another One...")
            pass
    if Res == [] or Res_Classes == []:
        return None, None
    return Res_Classes, Res


Ds = []
File_Path = ""
Pick = -1
while not(Pick == 0):
    while True:
        print("What You Wanna Do?")
        print("[2]Train Models")
        print("[1]Use Models")
        print("[0]Exit")
        Pick_Buf = input("Your Pick: ")
        try:
            Pick = int(Pick_Buf)
            if -1 < Pick < 3:
                break
            else:
                print("Number Out of Range, Try Again...")
        except ValueError:
            print("Only Int Input Allowed At This Stage, Try Again...")
            pass
    match Pick:
        case 2:
            Train_Type_Pick = -1
            while True:
                print("What Kind Of Model You Wanna Train?")
                print("[3]Word_2_Vec")
                print("[2]Doc_2_Vec")
                print("[1]Random Forest(W2V)")
                print("[0]Random Forest(D2V)")
                Pick_Buf = input("Your Pick: ")
                try:
                    Train_Type_Pick = int(Pick_Buf)
                    if -1 < Train_Type_Pick < 4:
                        if -1 < Train_Type_Pick < 2:
                            if Train_Type_Pick == 0:
                                Pick_Buf = input(
                                    "Have You Got D2V Model Or DataSet Processed by It?(y - yes, evrth else - no):")
                            else:
                                Pick_Buf = input(
                                    "Have You Got W2V Model Or DataSet Processed by It?(y - yes, evrth else - no):")
                            if Pick_Buf == "y":
                                break
                            else:
                                if Train_Type_Pick == 0:
                                    print(
                                        "D2V/D2V Processed DataSet, is Necessary for Random Forest Training Process, so Train it/Use it Before...")
                                else:
                                    print(
                                        "W2V/W2V Processed DataSet, is Necessary for Random Forest Training Process, so Train it/Use it Before...")
                        else:
                            break
                    else:
                        print("Number Out of Range, Try Again...")
                except ValueError:
                    print("Only Int Input Allowed At This Stage, Try Again...")
                    pass
            Data_Set_Pick = -1
            while True:
                Pick_Buf = input(
                    "You Want to Use Existing DataSet or Create One from Scratch?(0 - existing, 1 - new): ")
                try:
                    Data_Set_Pick = int(Pick_Buf)
                    if -1 < Data_Set_Pick < 2:
                        break
                    else:
                        print("Number Out of Range, Try Again...")
                except ValueError:
                    print("Only Int Input Allowed At This Stage, Try Again...")
                    pass
            while True:
                if Data_Set_Pick == 0:
                    File_Path = input("Enter Full Path to File With DataSet (must be json): ")
                else:
                    File_Path = input("Enter Full Path to File With Data (must be txt.gz): ")
                if os.path.exists(File_Path) and (
                        (".json" in File_Path and Data_Set_Pick == 0) or (
                        ".txt.gz" in File_Path and Data_Set_Pick == 1)):
                    break
                else:
                    print("File Doesn't Exist, or Has Wrong Extension, Try Again...")
            Save_Path = File_Path[:(File_Path.rfind('/') + 1)]
            Ds_Classes = []
            if Data_Set_Pick == 1:
                Texts = list(Read_Texts(File_Path))
                if 1 < Train_Type_Pick:
                    Ds_Save_Name = "Data_Set_W2V_D2V" + str(time.time()).replace(".", "") + ".json"
                    Ds = Create_Data_Set_W2V(Texts)
                    Save_Ds_To_File(Ds, Ds_Save_Name, Save_Path)
                    print("DataSet Saved at: " + Save_Path + Ds_Save_Name)
                else:
                    while True:
                        File_Path_Too = ""
                        if Train_Type_Pick == 0:
                            File_Path_Too = input("Enter Full Path To D2V File: ")
                        else:
                            File_Path_Too = input("Enter Full Path To W2V File: ")
                        if os.path.exists(File_Path):
                            break
                        else:
                            print("File Doesn't Exist, Try Again")
                    try:
                        if Train_Type_Pick == 0:
                            Model = doc2vec.Doc2Vec.load(File_Path_Too)
                            Ds, Ds_Classes = Texts_To_Vec(Texts, Model, Doc_2_Vec=True)
                            Ds_Save_Name = "Data_Set_Docs(Alt)_" + str(time.time()).replace(".", "") + ".json"
                            Save_Ds_To_File(Ds, Ds_Save_Name, Save_Path, Ds_Classes)
                            print("DataSet Saved at: " + Save_Path + Ds_Save_Name)
                        else:
                            Model = Word2Vec.load(File_Path_Too)
                            Ds, Ds_Classes = Texts_To_Vec(Texts, Model)
                            Ds_Save_Name = "Data_Set_Docs(W2V)_" + str(time.time()).replace(".", "") + ".json"
                            Save_Ds_To_File(Ds, Ds_Save_Name, Save_Path, Ds_Classes)
                            print("DataSet Saved at: " + Save_Path + Ds_Save_Name)
                    except ValueError:
                        print("Your Model Is Unusable In This Training, Try Another One...")
                        Train_Type_Pick = -1
                        pass
            else:
                if 1 < Train_Type_Pick:
                    Ds, Ds_Classes = Read_Ds(File_Path, True)
                else:
                    Ds = Read_Ds(File_Path)
                print("DataSet Uploaded!")
            if 1 < Train_Type_Pick:
                Tr_Test_Sep = round(len(Ds) * 0.9)
                Ds_Train = Ds[:Tr_Test_Sep]
                Ans_Train = Ds_Classes[:Tr_Test_Sep]
                Ds_Test = Ds[Tr_Test_Sep:]
                Ans_Test = Ds_Classes[Tr_Test_Sep:]
                New_Model = RandomForestClassifier(n_estimators=666, max_leaf_nodes=10, max_samples=15, random_state=0,
                                                   verbose=2, n_jobs=-1)
                New_Model.fit(Ds_Train, Ans_Train)
                Precision = New_Model.score(Ds_Test, Ans_Test)
                print("Created Forest Precision: " + str(Precision))
                Model_Save_Name = "Random_Forest_" + str(Precision) + ".pkl"
                _ = joblib.dump(New_Model, Save_Path + Model_Save_Name, compress=9)
                print("Random Forest Model Saved at: " + Save_Path + Model_Save_Name)
            elif Train_Type_Pick == 1:
                try:
                    New_Model = Word2Vec(Ds, min_count=0, workers=cpu_count())
                    W2V_Name = "W2V_Model_" + str(time.time()).replace(".", "")
                    New_Model.save(Save_Path + W2V_Name)
                    print("Word To Vec Model Saved at: " + Save_Path + W2V_Name)
                except ValueError:
                    print("Your DataSet is Unusable in This Training, Try Another One...")
                    pass
            elif Train_Type_Pick == 0:
                try:
                    Tag_Doc_Ds = list(tagged_document(Ds))
                    New_Model = doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)
                    New_Model.build_vocab(Tag_Doc_Ds)
                    New_Model.train(Tag_Doc_Ds, total_examples=New_Model.corpus_count, epochs=New_Model.epochs)
                    D2V_Name = "D2V_Model_" + str(time.time()).replace(".", "")
                    New_Model.save(Save_Path + D2V_Name)
                    print("Doc To Vec Model Saved at: " + Save_Path + D2V_Name)
                except ValueError:
                    print("Your DataSet is Unusable in This Training, Try Another One...")
                    pass
            Pick_Buf = input("Do You Wanna Stay?(y-yes, evrth else - no): ")
            if not (Pick_Buf == "y"):
                Pick = 0
        case 1:
            Use_Type_Pick = -1
            while True:
                print("What Kind Of Model You Wanna Use?")
                print("[1]Random Forest(W2V)")
                print("[0]Random Forest(D2V)")
                Pick_Buf = input("Your Pick: ")
                try:
                    Use_Type_Pick = int(Pick_Buf)
                    if -1 < Use_Type_Pick < 2:
                        if Use_Type_Pick == 0:
                            Pick_Buf = input("Have You Got D2V Model?(y - yes, evrth else - no):")
                        else:
                            Pick_Buf = input("Have You Got W2V Model?(y - yes, evrth else - no):")
                        if Pick_Buf == "y":
                            break
                        else:
                            if Use_Type_Pick == 0:
                                print(
                                    "D2V is Necessary for Random Forest Usage, so Train it In Different Main Menu Tab...")
                            else:
                                print(
                                    "W2V is Necessary for Random Forest Usage, so Train it In Different Main Menu Tab...")
                            Use_Type_Pick = -1
                            break
                    else:
                        print("Number Out of Range, Try Again...")
                except ValueError:
                    print("Only Int Input Allowed At This Stage, Try Again...")
                    pass
            if not (Use_Type_Pick == -1):
                while True:
                    Classes_Path = input("Enter Full Path To File With Classes(or -1 if You Don't Have/Need One): ")
                    if Classes_Path == "-1" or (os.path.exists(Classes_Path) and ".txt" in Classes_Path):
                        break
                    else:
                        print("File Doesn't Exist or Has Wrong Format, Try Again")
                Class_Idx, Class_Vals = None, None
                if not(Classes_Path == "-1"):
                    Class_Idx, Class_Vals = Upload_Classes(Classes_Path)
                File_Path_Too = ""
                while True:
                    File_Path_Too = input("Enter Full Path To Random Forest File: ")
                    if os.path.exists(File_Path_Too) and ".pkl" in File_Path_Too:
                        break
                    else:
                        print("File Doesn't Exist or Has Wrong Format, Try Again")
                while True:
                    if Use_Type_Pick == 0:
                        File_Path = input("Enter Full Path To D2V File: ")
                    else:
                        File_Path = input("Enter Full Path To W2V File: ")
                    if os.path.exists(File_Path):
                        break
                    else:
                        print("File Doesn't Exist, Try Again")
                Forest = joblib.load(File_Path_Too)
                Txt = input("Enter Text to Define Its Class Using Random Forest: ")
                if not (Txt == ""):
                    Text_Vecs = None
                    try:
                        if Use_Type_Pick == 0:
                            Model = doc2vec.Doc2Vec.load(File_Path)
                            Text_Vecs, _ = Texts_To_Vec([Text("", "", Txt)], Model, Doc_2_Vec=True)
                        else:
                            Model = Word2Vec.load(File_Path)
                            Text_Vecs, _ = Texts_To_Vec([Text("", "", Txt)], Model)
                        Res = str(Forest.predict(Text_Vecs))
                        Res = Res.replace("[", "")
                        Res = Res.replace("]", "")
                        if Class_Vals == None and Class_Idx == None:
                            print("Most Likely Class of Entered Text is: " + str(Res))
                        else:
                            print("Most Likely Class of Entered Text is: " + Class_Vals[Class_Idx.index(int(Res))])
                    except ValueError:
                        print("Entered Text Contains Unknown Words, Impossible to Work With It, Try Another One or Retrain Model/Use Different One...")
                        pass
            Pick_Buf = input("Do You Wanna Stay?(y-yes, evrth else - no): ")
            if not (Pick_Buf == "y"):
                Pick = 0

#text = "Парусная гонка Giraglia Rolex Cup пройдет в Средиземном море в 64-й раз. Победители соревнования, проводимого с 1953 года Yacht Club Italiano, помимо других призов традиционно получают в подарок часы от швейцарского бренда Rolex. Об этом сообщается в пресс-релизе, поступившем в редакцию «Ленты.ру» в среду, 8 мая. Rolex Yacht-Master 40 Фото: пресс-служба Mercury Соревнования будут проходить с 10 по 18 июня. Первый этап: ночной переход из Сан-Ремо в Сен-Тропе 10-11 июня (дистанция 50 морских миль — около 90 километров). Второй этап: серия прибрежных гонок в бухте Сен-Тропе с 11 по 14 июня. Финальный этап пройдет с 15 по 18 июня: оффшорная гонка по маршруту Сен-Тропе — Генуя (243 морских мили — 450 километров). Маршрут проходит через скалистый остров Джиралья к северу от Корсики и завершается в Генуе.Регата, с 1997 года проходящая при поддержке Rolex, считается одной из самых значительных яхтенных гонок в Средиземноморье. В этом году в ней ожидается участие трех российских экипажей."
#text = Clean_Text(text)
#arr_text = Clean_Text(text,Res_Type=1)
#arr_too_text = Clean_Text(text, Res_Type=2)
#print(text)
#print(arr_text)
#print(arr_too_text)
