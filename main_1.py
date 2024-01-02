import dataclasses
from yargy import Parser
from yargy.predicates import gram
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact
from yargy.relations import gnc_relation
from yargy import or_
from yargy import and_
from yargy.predicates import lte
from yargy.predicates import length_eq
from yargy.predicates import is_capitalized
from yargy.predicates import gte
from yargy.predicates import dictionary
from yargy import rule
from datetime import date
from dataclasses import dataclass
from typing import Optional
import gzip
from typing import Iterator
from tqdm import tqdm
import os
import json
import time


gnc = gnc_relation()

Name = fact(
    'Name',
    ['first', 'last']
)
class Name(Name):
    @property
    def as_string(self):
        return str(self.first) + " " + str(self.last)

FIRST = and_(
    gram('Name'),
    is_capitalized()
).interpretation(
    Name.first.inflected()
).match(gnc)

LAST = and_(
    gram('Surn'),
    is_capitalized()
).interpretation(
    Name.last.inflected()
).match(gnc)

Init = and_(length_eq(1), is_capitalized())

INIT = or_(
    rule(Init, "."),
    rule(Init, ".", Init, ".")
).interpretation(Name.first)

Middle_Name = and_(
    gram('NOUN'),
    is_capitalized()
)
#VV
NAME = or_(
    rule(
        FIRST,
        LAST
    ),
    rule(
        LAST,
        FIRST
    ),
    rule(FIRST, Middle_Name),
    rule(FIRST, Middle_Name, LAST),
    rule(LAST, FIRST, Middle_Name),
    rule(INIT, LAST),
    rule(LAST, INIT),
    rule(LAST),
    rule(FIRST)
).interpretation(
    Name
)

Date = fact(
    'Date of Birth',
    ['year', 'month', 'day']
)
class Date(Date):
    @property
    def as_string_list(self):
        Type = 0
        if not(self.year == None):
            Type = Type + 1
        if not(self.month == None):
            Type = Type + 3
        if not(self.day == None):
            Type = Type + 5
        match Type:
            case 9:
                try:
                    _ = date(self.year, self.month, self.day)
                    return [str(self.month), str(self.day), str(self.year)]
                except ValueError:
                    return [None]

            case 4:
                return [str(self.month), None, str(self.year)]
            case 8:
                try:
                    _ = date(2023, self.month, self.day)
                    return [str(self.month), str(self.day), None]
                except ValueError:
                    return [None]
            case 1:
                return [None, None, str(self.year)]
            case _:
                return [None]


MONTHS = {
    'январь': 1,
    'февраль': 2,
    'март': 3,
    'апрель': 4,
    'май': 5,
    'июнь': 6,
    'июль': 7,
    'август': 8,
    'сентябрь': 9,
    'октябрь': 10,
    'ноябрь': 11,
    'декабрь': 12
}

DAY = and_(
    gte(1),
    lte(31)
).interpretation(
    Date.day.custom(int)
)

MONTH = and_(
    gte(1),
    lte(12)
).interpretation(
    Date.month.custom(int)
)

YEAR = and_(
    gte(0),
    lte(2023)
).interpretation(
    Date.year.custom(int)
)

MONTH_NAME = dictionary(
    MONTHS
).interpretation(
    Date.month.normalized().custom(MONTHS.__getitem__)
)
#VV
DATE = or_(
    rule(YEAR, '-', MONTH, '-', DAY),
    rule(YEAR, '-', DAY, '-', MONTH),
    rule(DAY, '-', MONTH, '-', YEAR),
    rule(MONTH, '-', DAY, '-', YEAR),
    rule(DAY, MONTH_NAME, YEAR),
    rule(MONTH_NAME, YEAR),
    rule(DAY, MONTH_NAME),
    rule(YEAR)
).interpretation(Date)

Place = fact(
    'Birth Place',
    ['type', 'place']
)
class Place(Place):
    @property
    def as_string_list(self):
        Type = 0
        if not(self.type == None):
            Type = Type + 1
        if not(self.place == None):
            Type = Type + 3
        match Type:
            case 4:
                return [str(self.type), str(self.place)]
            case 3:
                return [None, str(self.place)]
            case _:
                return [None]


PLACES_TAGS = morph_pipeline([
    "село",
    "с.",
    "деревня",
    "д.",
    "поселок городского типа",
    "пгт.",
    "город",
    "г.",
    "поселок",
    "п."
])

Plc_Name = and_(
    gram('NOUN'),
    is_capitalized()
)

Plc_Name_Alt = and_(
    gram('ADJF'),
    is_capitalized()
)

PLACE_NAME = or_(
    rule(
        Plc_Name,
        "-",
        gram('PREP'),
        "-",
        Plc_Name
    ),
    rule(
        Plc_Name,
        "-",
        Plc_Name
    ),
    rule(
        Plc_Name_Alt,
        "-",
        Plc_Name
    ),
    rule(
        Plc_Name,
        Plc_Name
    ),
    rule(
        Plc_Name_Alt,
        Plc_Name
    ),
    rule(
        Plc_Name
    )
)
#VV
PLACE = or_(
    rule(
        PLACES_TAGS.interpretation(Place.type.inflected()).optional(),
        PLACE_NAME.interpretation(Place.place.inflected())
    ),
    rule(
        Plc_Name_Alt.interpretation(Place.place.inflected()),
        PLACES_TAGS.interpretation(Place.type.inflected()).optional()
    )
).interpretation(Place)

Entry = fact(
    "Entry",
    ["person", "birth_date", "birth_place"]
)

Connectors_Date = morph_pipeline([
    "родился",
    "родился в",
    "рождения"
])

Opt_Words_Date = morph_pipeline([
    "г.",
    "год",
    "месяц",
    "день",
    "утро",
    "вечер",
    "ночь"
])

#VV
ENTRY = or_(
    rule(
        #Tested
        Opt_Words_Date.optional(),
        gram('PNCT').optional(),
        DATE.interpretation(Entry.birth_date),
        gram('PNCT').optional(),
        Opt_Words_Date.optional(),
        "в",
        gram('ADJF').optional().repeatable(),
        PLACE.interpretation(Entry.birth_place),
        gram('PNCT').optional(),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        NAME.interpretation(Entry.person)
    ),
    rule(
        #Tested
        PLACE.interpretation(Entry.birth_place),
        gram('PREP').optional(),
        gram('ADJF').optional().repeatable(),
        Opt_Words_Date.optional(),
        gram('PREP').optional(),
        DATE.interpretation(Entry.birth_date),
        Opt_Words_Date.optional(),
        gram('PNCT').optional(),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        NAME.interpretation(Entry.person)
    ),
    rule(
        #Tested
        PLACE.interpretation(Entry.birth_place),
        gram('PREP').optional(),
        gram('NPRO').optional().repeatable(),
        gram('ADJF').optional().repeatable(),
        Opt_Words_Date.optional(),
        gram('PNCT').optional(),
        DATE.interpretation(Entry.birth_date),
        gram('PNCT').optional(),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        NAME.interpretation(Entry.person)
    ),
    rule(
        #Tested
        NAME.interpretation(Entry.person),
        gram('PNCT').optional(),
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        gram('PNCT').optional(),
        Connectors_Date,
        DATE.interpretation(Entry.birth_date),
        Opt_Words_Date.optional(),
        "в",
        gram('ADJF').optional().repeatable(),
        PLACE.interpretation(Entry.birth_place)
    ),
    rule(
        #Tested
        NAME.interpretation(Entry.person),
        gram('PNCT').optional(),
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        gram('PNCT').optional(),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        PLACE.interpretation(Entry.birth_place),
        gram('PNCT').optional(),
        DATE.interpretation(Entry.birth_date)
    ),
    rule(
        #Tested
        PLACE.interpretation(Entry.birth_place),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        gram('PNCT').optional(),
        NAME.interpretation(Entry.person)
    ),
    rule(
        DATE.interpretation(Entry.birth_date),
        Opt_Words_Date,
        gram('PNCT').optional(),
        gram('PREP').optional(),
        gram('ADJF').optional().repeatable(),
        Opt_Words_Date,
        Connectors_Date,
        NAME.interpretation(Entry.person)
    ),
    rule(
        #Tested
        Opt_Words_Date.optional(),
        gram('PNCT').optional(),
        DATE.interpretation(Entry.birth_date),
        gram('PNCT').optional(),
        Opt_Words_Date.optional(),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        NAME.interpretation(Entry.person)
    ),
    rule(
        #Tested
        NAME.interpretation(Entry.person),
        gram('PNCT').optional(),
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        gram('PNCT').optional(),
        Connectors_Date,
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        PLACE.interpretation(Entry.birth_place)
    ),
    rule(
        DATE.interpretation(Entry.birth_date),
        gram('PNCT').optional(),
        gram('NOUN').optional().repeatable(),
        gram('VERB').optional(),
        Opt_Words_Date,
        Connectors_Date,
        NAME.interpretation(Entry.person)
    ),
    rule(
        NAME.interpretation(Entry.person),
        gram('PNCT').optional(),
        gram('NOUN').optional(),
        PLACE.interpretation(Entry.birth_place),
        DATE.interpretation(Entry.birth_date),
        Opt_Words_Date,
        Connectors_Date
    ),
    rule(
        #Tested
        NAME.interpretation(Entry.person),
        gram('PNCT').optional(),
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        gram('PNCT').optional(),
        DATE.interpretation(Entry.birth_date),
        Opt_Words_Date,
        Connectors_Date
    ),
    rule(
        #Tested
        NAME.interpretation(Entry.person),
        gram('PNCT').optional(),
        gram('ADJF').optional().repeatable(),
        gram('NOUN').optional().repeatable(),
        gram('PNCT').optional(),
        Connectors_Date,
        DATE.interpretation(Entry.birth_date)
    ),
    rule(
        #Tested
        NAME.interpretation(Entry.person)
    )
).interpretation(Entry)


@dataclass
class Entry:
    name: str
    birth_date: Optional[str]
    birth_place: Optional[str]

@dataclass
class Text:
    label: str
    title: str
    text: str


def Read_Texts(fn: str) -> Iterator[Text]:
    with gzip.open(fn, "rt", encoding="utf-8") as File:
        for Line in File:
            yield Text(*Line.strip().split("\t"))


def Save_Entry_To_File(Entries_List, File_Name, File_Path):
    with open(File_Path + File_Name, "w+") as File:
        File.write(
            f"{json.dumps({'data' : Entries_List}, default=dataclasses.asdict, indent=4)}\n"
        )
        File.close()


parser = Parser(ENTRY)
Pars_Res = []
File_Path = ""
while True:
    File_Path = input("Enter Full Path To File + File Name with Extension: ")
    if os.path.exists(File_Path):
        break
    else:
        print("File Doesn't Exist, Try Again")

Save_Path = File_Path[:(File_Path.rfind('/') + 1)]
Save_Name = "Res_" + str(time.time()).replace(".", "") + ".json"
Texts = list(Read_Texts(File_Path))
for text in tqdm(Texts):
    #neccessary move, since yargy don't want to work with any punctuation, even though I use gram('PNCT')
    Txt = text.text.replace("-", "")
    Txt = Txt.replace("—", "")
    Txt = Txt.replace(",", "")
    Txt = Txt.replace("(", "")
    Txt = Txt.replace(")", "")
    Txt = Txt.replace(":", "")
    try:
        for match in parser.findall(text.text):
            if not (match.fact.birth_date == None) and not (match.fact.birth_place == None):
                Pars_Res.append(Entry(match.fact.person.as_string, match.fact.birth_date.as_string_list, match.fact.birth_place.as_string_list))
            elif not(match.fact.birth_date == None):
                Pars_Res.append(Entry(match.fact.person.as_string, match.fact.birth_date.as_string_list, None))
            elif not(match.fact.birth_place == None):
                Pars_Res.append(Entry(match.fact.person.as_string, None, match.fact.birth_place.as_string_list))
            else:
                Pars_Res.append(Entry(match.fact.person.as_string, None, None))
    except ValueError:
        print("Something wet wrong, Error: ")
        print(ValueError)
        pass
print("Result:")
print(Pars_Res)
Save_Entry_To_File(Pars_Res, Save_Name, Save_Path)
print("Result Saved at: " + Save_Path + Save_Name)

#----------Testing_Name-------------(Result: Working Perfectly)
#text_nm = '''Кобейн К., Курт Кобейн, Курт Дональд Кобейн, К. Кобейн, К.Д. Кобейн'''
#parser = Parser(NAME)
#for match in parser.findall(text_nm):
#    print(match.fact.as_string)


#----------Testing_Date-------------(Result: Working Perfectly)
#text_too = '''8 января 2014 года, 2018-12-01, 23 июня, 1976, в марте 2010, 30 февраля 1337'''
#parser = Parser(DATE)
#for match in parser.findall(text_too):
#    print(match.fact.as_string_list)


#----------Testing_Places-------------(Result: Working Fine, Except 'Ачи-Су', Since It's Not Defined as a 'NOUN' or 'ADJF' in pymorphy2)
#text_too = '''в деревне Грязь, поселок городского типа Ачи-Су, в городе Великом Новгороде, неподалеку от села Республика Алтай, в непосредственной близости от г. Ростова-на-Дону'''
#parser = Parser(PLACE)
#for match in parser.findall(text_too):
#    print(match.fact.as_string_list)


#----------Testing_Etries-------------(Result: Working Fine, Have Some Problems With ',' and Some Names/Nouns (Doesn't Identify Them))
#text = '''23 июля 1976 года в поселке городского типа Адлер родился выдающийся советский ученый Аристарх Столыпин, В небольшом украинском селе Отвал в ночь на 12 июня 1900 года родился Вольдемар Дзержинский, В невероятном городе Сингапур в этот день 30 июня родился Спартак Ульянов, Казимир Измайлов родился 28 июня 1910 года в поселке городского типа Совок, Ярополк Аскольдович родился в замечательной швейцарской деревне Гиммельвальд 23 октября, В сером и скучном поселке Москва родился знаменитый рабочий Пересвет Святоборович Кадыров, В этот день 24 августа родился Пафнутий Сталин, Весимир Пугин родился в прекрасном городе Люксембурге, Добрыня Ленин родился 25 апреля 2020, Акакий Черненко'''
#parser = Parser(ENTRY)
#Pars_Res = []
#text = text.replace(",", "")
#text = text.replace("-", "")
#text = text.replace(":", "")
#text = text.replace("(", "")
#text = text.replace(")", "")
#print(text)
#for match in parser.findall(text):
#    if not(match.fact.birth_date == None) and not(match.fact.birth_place == None):
#        Pars_Res.append(Entry(match.fact.person.as_string, match.fact.birth_date.as_string_list, match.fact.birth_place.as_string_list))
#    elif not(match.fact.birth_date == None):
#        Pars_Res.append(Entry(match.fact.person.as_string, match.fact.birth_date.as_string_list, None))
#    elif not(match.fact.birth_place == None):
#        Pars_Res.append(Entry(match.fact.person.as_string, None, match.fact.birth_place.as_string_list))
#    else:
#        Pars_Res.append(Entry(match.fact.person.as_string, None, None))
#    print(match.fact)
#print(Pars_Res)
