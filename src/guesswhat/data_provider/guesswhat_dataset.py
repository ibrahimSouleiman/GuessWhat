import gzip
import json
import copy
import os
import nltk
import time
import numpy as np

from collections import Counter
from matplotlib import pyplot as plt

from PIL import ImageFont, ImageDraw
from PIL import Image as PImage
from nltk.tokenize import TweetTokenizer
from nltk import WordNetLemmatizer

from generic.data_provider.dataset import AbstractDataset
from draw.Generate_Category import get_category
# TODO find a cleaner way!
try:
    import cocoapi.PythonAPI.pycocotools.mask as cocoapi
    use_coco = True
except ImportError:
    print("Coco API was not detected - advanced segmentation features cannot be used")
    use_coco = False
    pass




class Game:

    def __init__(self, id, object_id, image, objects, qas , status, which_set, image_builder, crop_builder,lemmatizer,all_img_bbox,all_img_describtion):
        self.dialogue_id = id
        self.object_id = object_id
        
        compteur = 0

        self.image = Image(id=image["id"],
                           width=image["width"],
                           height=image["height"],
                           url=image["coco_url"],
                           description=image["description"],
                           which_set=which_set,
                           image_builder=image_builder)

    

        self.objects = []
        self.all_category = []
        for o in objects:
            self.all_category.append(o['category'])
            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=o['area'],
                             segment=o['segment'],
                             crop_builder=crop_builder,
                             which_set=which_set,
                             image=self.image)

            

            self.objects.append(new_obj)

            if o['id'] == object_id:
                self.object = new_obj  # Keep ref on the object to find
                # all_img_bbox[image["id"] ]= o['bbox']    
                # print("Select=",o['category'])
            else:
                pass
                # print(o['category'])     

        # print("-----------------------")
        compteur += 1
        if compteur == 15:
            exit()


        # all_img_describtion[image["id"]] = image["description"]

        self.question_ids = [qa['id'] for qa in qas]
        
        self.questions = [qa['question'] for qa in qas]
        
        self.answers = [qa['answer'] for qa in qas]

        self.status = status


        self.last_question = ["unk" for i in range(1)]
        
        self.all_last_question = [self.last_question for i in range(6)]


    def show(self, img_raw_dir, display_index=False, display_mask=False):
            image_path = os.path.join(img_raw_dir, self.image.filename)

            img = PImage.open(image_path)
            draw = ImageDraw.Draw(img)

            for i, obj in enumerate(self.objects):
                if display_index:
                    draw.text((obj.bbox.x_center, self.image.height-obj.bbox.y_center), str(i))
                if display_mask:
                    print("Show mask: Not yet implemented... sry")

            img.show()



class Word:
    """
    classe qui contient pour chaque mot , son lemme sa parti de discourt

    
    """

    def __init__(self,word,lemme=False,pos=False,lemmatizer=None):

        self.word = word
        self.lemme = ""
        self.pos = ""
        self.lemmetizer = lemmatizer
        if self.lemmetizer != None:
        
            if lemme :
                self.lemme = self.lemmetizer.lemmatize(word)
            
        if pos:
            self.pos = nltk.pos_tag([word])

    def get_word(self):
        return self.word

    def get_lemme(self):
        return self.lemme
    
    def get_pos(self):
        return self.pos

    def get_all(self):
        return self.word,self.lemme,self.pos
    





class Image:
    def __init__(self, id, width, height, url,description, which_set, image_builder=None):
        self.id = id
        self.width = width
        self.height = height
        self.url = url
        self.description = description

        self.image_loader = None
        if image_builder is not None:
            self.filename = "{}.jpg".format(id)
            self.image_loader = image_builder.build(id, which_set=which_set, filename=self.filename, optional=False)


    def get_idimage(self):
        return self.id

    def get_image(self, **kwargs):
        if self.image_loader is not None:
            return self.image_loader.get_image(**kwargs)
        else:
            return None




class Bbox:
    def __init__(self, bbox, im_width, im_height):
        # Retrieve features (COCO format)
        self.x_width = bbox[2]
        self.y_height = bbox[3]

        self.x_left = bbox[0]
        self.x_right = self.x_left + self.x_width

        self.y_upper = im_height - bbox[1]
        self.y_lower = self.y_upper - self.y_height

        self.x_center = self.x_left + 0.5*self.x_width
        self.y_center = self.y_lower + 0.5*self.y_height

        self.coco_bbox = bbox

    def __str__(self):
        return "center : {0:5.2f}/{1:5.2f} - size: {2:5.2f}/{3:5.2f}"\
            .format(self.x_center, self.y_center, self.x_width, self.y_height)


class Object:
    def __init__(self, id, category, category_id, bbox, area, segment, crop_builder, image, which_set):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segment = segment
 
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        # self.rle_mask = None
        # if use_coco:
        #     self.rle_mask = cocoapi.frPyObjects(self.segment,
        #                                         h=image.height,
        #                                         w=image.width)

        if crop_builder is not None:
            filename = "{}.jpg".format(image.id)
            self.crop_loader = crop_builder.build(id, filename=filename, which_set=which_set, bbox=bbox)            # print("Image_id=",image.id)
            # print("Bbox=",bbox)
            # exit()
    # def get_mask(self):
    #     assert self.rle_mask is not None, "Mask option are not available, please compile and link cocoapi (cf. cocoapi/PythonAPI/setup.py)"
    #     return cocoapi.decode(self.rle_mask)

    def get_idObject(self):
        return self.id

    def get_crop(self, **kwargs):
        assert self.crop_loader is not None, "Invalid crop loader"
        return self.crop_loader.get_image(**kwargs)




class Dataset(AbstractDataset):
    """Loads the dataset."""
    def __init__(self, folder, which_set, image_builder=None, crop_builder=None,all_img_bbox=None,all_img_describtion=None):
        filegues = '{}/guesswhat_{}2014.jsonl.gz'.format(folder,which_set)
        file_description = '{}/captions_{}2014.json'.format(folder, which_set)

        games = []
        nb_pass = 0
        nb_erreur = 0

        self.set = which_set
        self.lemmas = WordNetLemmatizer()

        self.maxlength_question = 0
        self.count_questions = {}
        self.indice = 0
        self.total = 0

        tokenizer = TweetTokenizer(preserve_case=True)


        t1 = time.time()

        all_size = []

        with gzip.open(filegues) as f:
            for i,line in enumerate(f):
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))
                # try : 
                    # print(game["image"])
                    # print("Dans try")
                    # print(game["id"],type(game["id"]))
                nb_pass += 1
                g = Game(id=game['id'],
                        object_id=game['object_id'],
                        objects=game['objects'],
                        qas=game['qas'],
                        image=game['image'],
                        status=game['status'],
                        which_set=which_set,
                        image_builder=image_builder,
                        crop_builder=crop_builder,
                        lemmatizer = self.lemmas,
                        all_img_bbox = all_img_bbox,
                        all_img_describtion = all_img_describtion
                        )
                question_length = len(g.questions)
                

                self.count_questions[len(self.count_questions)] = question_length
                self.total += question_length 
                
                for question in g.questions:
                    words = question.split()
                 
                    all_size.append(len(words))

            

                if self.maxlength_question < question_length: self.maxlength_question = question_length

                games.append(g)
                



                #     # exit()
                # except TypeError:
                #     print("error to create dataset")
                #     nb_erreur += 1


                # print("NP_pass = {} , nb_erreur = {} ".format(nb_erreur,nb_pass)               
               
                if len(games) > 50: break
                # if  len(games) > 5000: 

           
                #  break



        print(" Max Length of Question = {} , total_question = {}, nb_parties = {} | {}".format(self.maxlength_question,self.total,len(self.count_questions),which_set))


       


        super(Dataset, self).__init__(games)

class OracleDataset(AbstractDataset):
    """
    Each game contains a single question answer pair
    """
    def __init__(self, dataset):
        old_games = dataset.get_data()
        new_games = []
        self.compteur = 0
        self.all_category = []
        self.categories_questions = {0:0,1:0,2:0,3:0}

        self.length_question = {}
        
        for i,g in enumerate(old_games):
            new_games += self.split(g)   
        # print(" Nb question = ",format(self.compteur))
        # exit()
        # print(" Guess_dataset | Lemme different = {} ",format(self.compteur))

        # for i in range(1000):
        #     print("Question = ",new_games[i].questions)

        self.unique_category = np.unique(np.asarray(self.all_category))
        print("Distrubution of category = ",self.categories_questions)
        # print(Counter(self.all_category))
        # print("len=",len(self.unique_category))
        # print("Conteur = ",self.compteur)

        # print(self.length_question)
        # exit()
        
        super(OracleDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set,all_category=None, image_builder=None, crop_builder=None,all_img_bbox=None,all_img_describtion=None):
        
        print("#################### {} ##########################".format(which_set))

        return cls(Dataset(folder, which_set, image_builder, crop_builder,all_img_bbox,all_img_describtion))

    def split(self, game):

        games = []
        for i, q, a in zip(game.question_ids, game.questions, game.answers):
            # print("****** GuessWhat | oracleDataSet q={}".format(q))
            # if( self.compteur == 10):
            #     exit()

            wpt = TweetTokenizer(preserve_case=False)
            # tokens = [ token for token in wpt.tokenize(q)]
            # lemme = [self.lemmas.lemmatize(token) for token in tokens]
            categorie_question = get_category(q,wpt)
            self.categories_questions[categorie_question] += 1  
            
            new_game = copy.copy(game)
            new_game.questions = [q]
            # print("question = {} ,taille = {} ".format(q,len(q.split(" "))))

            # try:
            #     self.length_question[len(q.split(" "))] += 1
            # except KeyError:
            #     self.length_question[len(q.split(" "))] = 0


            new_game.question_ids = [i]
            new_game.answers = [a]
            # print(q)
            self.compteur += 1 

            for category in game.all_category:
                self.all_category.append(category)

            
            last_question =  self.add_question(game.all_last_question,q,a)

            # print("All question = {} ".format(game.all_last_question))

            # exit()

            new_game.all_last_question = last_question
            # print("Image_id={}, dialogue_id,={}, question_id={}, question={},last_question={}".format(new_game.image.id,new_game.dialogue_id,i,q,last_question))

            games.append(new_game)

        
        return games


    def add_question(self,last_questions,question,answer):


        before = ""
        # print("In = {} ",last_questions)

        i = 0
        nb_question = [i+1 for s in last_questions if s[0]!="unk"]

        list_copy = last_questions.copy()

        nb_question = int(np.sum(nb_question)) + 1
        # print("Nb_question = {},question={}".format(nb_question,question))
        for i in  reversed(range(6)):
            if i == 5:
                last_questions[i] = [ [answer],[question]]
                # print("i={},{}".format(i,last_questions[i]))

            elif  nb_question > 0 and i > 0:
                # print("i={},{}".format(i,list_copy[i+1]))
                last_questions[i] =  list_copy[i+1]

            nb_question -= 1


        return last_questions



class CropDataset(AbstractDataset):
    """    Each game contains no question/answers but a new object
    """
    def __init__(self, dataset):
        old_games = dataset.get_data()
        new_games = []
        for g in old_games:
            new_games += self.split(g)
        super(CropDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set, image_builder=None, crop_builder=None):
        return cls(Dataset(folder, which_set, image_builder, crop_builder))

    def split(self, game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)
            new_game.questions = []
            new_game.question_ids = []
            new_game.answers = []
            new_game.object_id = obj.id

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)
            new_game.image.id = obj.id

            games.append(new_game)

        return games


def dump_samples_into_dataset(data, save_path, tokenizer, name="model"):

    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for id_game, d in enumerate(data):
            dialogue = d["dialogue"]
            game = d["game"]
            object_id = d["object_id"]
            success = d["success"]
            prob_objects = d["prob_objects"]
            guess_object_id = d["guess_object_id"]

            sample = {}

            qas = []
            start  = 1
            for k, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                                word == tokenizer.no_token or \
                                word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:k - 1])
                    a = tokenizer.decode([dialogue[k]])

                    prob_obj = list(prob_objects[len(qas),:len(game.objects)])
                    prob_obj = [str(round(p,3)) for p in prob_obj] # decimal are not supported y default in json encoder

                    qas.append({"question": q,
                                "answer": a[1:-1],
                                "id":0,
                                "p": prob_obj})

                    start = k + 1

            sample["id"] = id_game
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment" : [], #no segment to avoid making the file to big
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["guess_object_id"] = guess_object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')
