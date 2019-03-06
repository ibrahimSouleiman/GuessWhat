import gzip
import json
import copy
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creating data description..')

    parser.add_argument("-data_type", type=str, help="data train /val/test")

    args = parser.parse_args()


    file = '{}/guesswhat.{}.jsonl.gz'.format("data", args.data_type)

    file_descriptionT = '{}/captions_{}2014.json'.format("data/annotations", "train")

    file_descriptionV = '{}/captions_{}2014.json'.format("data/annotations", "val")
   
    file_output= '{}/guesswhat_{}2014.jsonl'.format("data", args.data_type)


    games = []
    list_games = []
    description = {}
    description_notFound = []


    with  open(file_descriptionT)  as fd:
        game = json.load(fd)
        
        for line in game["annotations"]:

            description[line["image_id"]] = line["caption"]
       
    with  open(file_descriptionV)  as fd:
        game = json.load(fd)
        
        for line in game["annotations"]:

            description[line["image_id"]] = line["caption"]
     
    n = 0
    with gzip.open(file) as f:
        for line in f:
            line = line.decode("utf-8")
            game = json.loads(line.strip('\n'))
        
            link = game["image"]["id"]
            id_img = int(link)

            try:
                game["image"]["description"]  = description[id_img]
                n += 1
            except KeyError:
                game["image"]["description"]  = " Not Found "
                description_notFound.append(id_img)

            # with open(file_output,"a") as new_file:
            #     new_file.write("{%s}" % ",\n ".join(json.dumps(game)))
            
            list_games.append(game)

            # if n == 3:
            #     break
            # else:
            #     n += 1
        print("Nb description added = {} ".format(n))
        # Nb description train = 113221 
        # Nb description valid = 23739  
        # Nb description test = 23785  
        open(file_output,'w').write("%s" % "\n ".join(json.dumps(e) for e in list_games))

    # with open(file_output,"a") as new_file:
    #     #json.dump(list_games, new_file)

                

    print("Type = {} , not_found_description = {} , description_length = {}".format(args.data_type,len(description_notFound),len(description)))


