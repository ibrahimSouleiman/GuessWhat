import gzip
import json
import copy
import os
import json_lines

file = '{}/guesswhat.{}.jsonl.gz'.format("data", "train")

file_descriptionT = '{}/captions_{}2014.json'.format("data/annotations_Caption", "train")

file_descriptionV = '{}/captions_{}2014.json'.format("data/annotations_Caption", "val")

file_new = 'guesswhat.train.new.jsonl.gz'
file_new = 'data/guesswhatDescription.jsonl.gz'


games = []

description = {}
description_notFound = []


# with  open(file_descriptionT)  as fd:
#     game = json.load(fd)
    
#     for line in game["annotations"]:

#         description[line["image_id"]] = line["caption"]
    # print(description[460809])
    # exit()
# with  open(file_descriptionV)  as fd:
#     game = json.load(fd)
    
#     for line in game["annotations"]:

#         description[line["image_id"]] = line["caption"]
#     # print(description[460809])
#     # exit()

# # self.set = which_set
# n = 0


with gzip.open(file_new) as f:
    print("before for ..")
    for line in f:
        print("in for ..")

        line = line.decode("utf-8")
        print("after decode ..")

        game = json.loads(line.strip('\n'))
        print("after game ..")

        for img in game:
                print("in loop game ..")

                print(img["image"])
                exit()



# with open(file_new) as f:
#     game = json.load(f)
#     print(game)
    # for line in f:
    #     print(line)
    #     game = json.loads(line)
    #     print(game)
        


    # for line in f:
    #     # line = line.decode("utf-8")
    #     game = json.load(line)
    #     print(game)

        # with open("newfile.json","a") as new_file:
        #     # print(game["image"]["coco_url"])
        #     link = game["image"]["id"]
        #     id_img = int(link)
        #     # print(id_img)

        #     try:
        #         game["image"]["description"]  = description[id_img]
        #     except KeyError:
        #         # print("Not FOund")
        #         game["image"]["description"]  = " Not Found "

        #         description_notFound.append(id_img)

        #     # print("creation new file ...")
        #     json.dump(game, new_file)
             

# print(len(description_notFound),len(description))


#         # g = Game(id=game['id'],
#         #           object_id=game['object_id'],
#         #           objects=game['objects'],
#         #           qas=game['qas'],
#         #           image=game['image'],
#         #           status=game['status'],
#         #           which_set=which_set,
#         #           image_builder=image_builder,
#         #           crop_builder=crop_builder)
#         # description[id_img]