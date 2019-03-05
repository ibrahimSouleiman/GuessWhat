import gzip
import json
import copy
import os

file = '{}/guesswhat.{}.jsonl.gz'.format("data", "train")

file_descriptionT = '{}/captions_{}2014.json'.format("data/annotations_Caption", "train")

file_descriptionV = '{}/captions_{}2014.json'.format("data/annotations_Caption", "val")

games = []
list_games = []
description = {}
description_notFound = []


with  open(file_descriptionT)  as fd:
    game = json.load(fd)
    
    for line in game["annotations"]:

        description[line["image_id"]] = line["caption"]
    # print(description[460809])
    # exit()
with  open(file_descriptionV)  as fd:
    game = json.load(fd)
    
    for line in game["annotations"]:

        description[line["image_id"]] = line["caption"]
    # print(description[460809])
    # exit()

# self.set = which_set

n = 0
with gzip.open(file) as f:
    for line in f:
        line = line.decode("utf-8")
        game = json.loads(line.strip('\n'))
    
        # print(game["image"]["coco_url"])
        link = game["image"]["id"]
        id_img = int(link)
        # print(id_img)

        try:
            game["image"]["description"]  = description[id_img]
        except KeyError:
            game["image"]["description"]  = " Not Found "
            description_notFound.append(id_img)
        list_games.append(game)
        # print("creation new file ...")

        # if n == 3:
        #     break
        # else:
        #     n += 1


with open("data/guesswhatDescription.jsonl","a") as new_file:
    json.dump(list_games, new_file)

             

print(len(description_notFound),len(description))


        # g = Game(id=game['id'],
        #           object_id=game['object_id'],
        #           objects=game['objects'],
        #           qas=game['qas'],
        #           image=game['image'],
        #           status=game['status'],
        #           which_set=which_set,
        #           image_builder=image_builder,
        #           crop_builder=crop_builder)
        # description[id_img]