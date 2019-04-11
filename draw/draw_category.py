
import matplotlib.pyplot as plt
import pandas as pd

train = {'person':[ 1761187], 'car':[ 303517], 'book':[ 237638], 'chair':[ 229932], 'bottle':[ 179442], 'cup':[ 138509], 'bowl':[ 110176], 'dining table':[ 104251], 'boat':[ 76270], 'handbag':[ 75705], 'traffic light':[ 73951], 'sheep':[ 70477], 'truck':[ 68977], 'carrot':[ 68593], 'potted plant':[ 64375], 'bench':[ 63809], 'banana':[ 61847], 'broccoli':[ 61080], 'donut':[ 57612], 'bird':[ 56074], 'cow':[ 54785], 'knife':[ 54133], 'backpack':[ 53533], 'umbrella':[ 51783], 'vase':[ 50729], 'couch':[ 46883], 'motorcycle':[ 46284], 'orange':[ 46169], 'tv':[ 45951], 'remote':[ 44523], 'spoon':[ 44120], 'cake':[ 43416], 'apple':[ 42494], 'bicycle':[ 42297], 'bus':[ 41292], 'kite':[ 39592], 'wine glass':[ 39290], 'suitcase':[ 38669], 'cell phone':[ 38598], 'sink':[ 37745], 'laptop':[ 36591], 'fork':[ 36253], 'horse':[ 36166], 'tie':[ 35746], 'skis':[ 35331], 'pizza':[ 34385], 'clock':[ 34087], 'teddy bear':[ 33812], 'sports ball':[ 31471], 'elephant':[ 31241], 'sandwich':[ 30433], 'skateboard':[ 29393], 'oven':[ 27706], 'surfboard':[ 26969], 'airplane':[ 25729], 'tennis racket':[ 25602], 'dog':[ 25463], 'baseball glove':[ 24960], 'zebra':[ 24435], 'refrigerator':[ 23543], 'baseball bat':[ 23301], 'keyboard':[ 21937], 'hot dog':[ 20185], 'bed':[ 19787], 'train':[ 19416], 'mouse':[ 18305], 'cat':[ 17877], 'giraffe':[ 16976], 'toilet':[ 15434], 'microwave':[ 14339], 'frisbee':[ 13106], 'toothbrush':[ 12397], 'snowboard':[ 11773], 'scissors':[ 10988], 'stop sign':[ 9461], 'fire hydrant':[ 8734], 'parking meter':[ 7882], 'toaster':[ 2397], 'bear':[ 1533], 'hair drier':[ 1167]}
valid = {'person':[ 347547], 'car':[ 67251], 'book':[ 47512], 'chair':[ 44443], 'bottle':[ 41404], 'cup':[ 30701], 'bowl':[ 24460], 'dining table':[ 20982], 'boat':[ 15598], 'sheep':[ 15452], 'banana':[ 15276], 'handbag':[ 15131], 'carrot':[ 15047], 'traffic light':[ 14266], 'truck':[ 13717], 'bench':[ 13164], 'potted plant':[ 13031], 'broccoli':[ 11922], 'backpack':[ 11551], 'knife':[ 11546], 'cow':[ 11386], 'donut':[ 11190], 'orange':[ 11034], 'umbrella':[ 10923], 'tv':[ 10736], 'bird':[ 10725], 'couch':[ 10030], 'suitcase':[ 9681], 'bicycle':[ 9674], 'spoon':[ 9598], 'vase':[ 9424], 'horse':[ 9321], 'cake':[ 9096], 'remote':[ 8617], 'fork':[ 8507], 'bus':[ 8506], 'sink':[ 8503], 'cell phone':[ 8209], 'skis':[ 7855], 'motorcycle':[ 7624], 'pizza':[ 7526], 'wine glass':[ 7510], 'apple':[ 7252], 'kite':[ 7157], 'clock':[ 7125], 'laptop':[ 6939], 'tie':[ 6867], 'elephant':[ 6854], 'skateboard':[ 6262], 'surfboard':[ 6261], 'sports ball':[ 5872], 'oven':[ 5809], 'dog':[ 5787], 'sandwich':[ 5634], 'zebra':[ 5527], 'tennis racket':[ 5507], 'keyboard':[ 5186], 'baseball bat':[ 4684], 'teddy bear':[ 4665], 'hot dog':[ 4628], 'refrigerator':[ 4621], 'baseball glove':[ 4407], 'bed':[ 4311], 'train':[ 4282], 'airplane':[ 4039], 'cat':[ 4007], 'mouse':[ 3726], 'toothbrush':[ 3131], 'giraffe':[ 2990], 'frisbee':[ 2926], 'microwave':[ 2881], 'toilet':[ 2510], 'snowboard':[ 2463], 'fire hydrant':[ 1860], 'stop sign':[ 1688], 'parking meter':[ 1533], 'scissors':[ 1481], 'toaster':[ 396], 'bear':[ 359], 'hair drier':[ 127]}
test = {'person':[ 364378], 'car':[ 67190], 'book':[ 56996], 'chair':[ 47388], 'bottle':[ 36927], 'cup':[ 27666], 'bowl':[ 24708], 'dining table':[ 20705], 'traffic light':[ 16804], 'handbag':[ 16075], 'boat':[ 15824], 'sheep':[ 15818], 'banana':[ 15540], 'carrot':[ 15198], 'cow':[ 14379], 'bird':[ 13895], 'truck':[ 13836], 'potted plant':[ 13122], 'bench':[ 12777], 'broccoli':[ 12241], 'backpack':[ 12142], 'cake':[ 11975], 'donut':[ 11730], 'knife':[ 11398], 'tv':[ 10560], 'umbrella':[ 10226], 'suitcase':[ 9832], 'vase':[ 9772], 'couch':[ 9672], 'bus':[ 9127], 'motorcycle':[ 9050], 'horse':[ 9031], 'spoon':[ 8896], 'kite':[ 8888], 'orange':[ 8738], 'apple':[ 8501], 'bicycle':[ 8352], 'skis':[ 8316], 'remote':[ 8142], 'sink':[ 7929], 'cell phone':[ 7431], 'fork':[ 7220], 'laptop':[ 7045], 'pizza':[ 6867], 'tie':[ 6830], 'wine glass':[ 6824], 'sports ball':[ 6599], 'clock':[ 6590], 'teddy bear':[ 6372], 'sandwich':[ 6316], 'skateboard':[ 6065], 'dog':[ 6020], 'oven':[ 5801], 'elephant':[ 5761], 'zebra':[ 5469], 'baseball bat':[ 5346], 'surfboard':[ 5108], 'bed':[ 4989], 'tennis racket':[ 4976], 'baseball glove':[ 4948], 'airplane':[ 4824], 'keyboard':[ 4528], 'refrigerator':[ 4407], 'train':[ 4319], 'hot dog':[ 4247], 'snowboard':[ 4173], 'cat':[ 3927], 'giraffe':[ 3733], 'mouse':[ 3589], 'toilet':[ 3017], 'frisbee':[ 2912], 'microwave':[ 2804], 'toothbrush':[ 2047], 'parking meter':[ 1794], 'scissors':[ 1663], 'fire hydrant':[ 1321], 'stop sign':[ 1279], 'bear':[ 400], 'toaster':[ 388], 'hair drier':[ 333]}


# # plt.subplot(3],1],1)
# print(range(len(train)))
# taille = (6],4)

# plt.figure(figsize=(50],6)], dpi=80)

# plt.bar(range(1],len(train)+1)],list(train.values())],align='center')
# # plt.xticks(range(len(train))], list(train.keys()))
# print(list(train.keys()))
# plt.legend(list(train.keys()))
# plt.axes(1],80)
# plt.show()
df = pd.DataFrame.from_dict(data = train,orient='index',index=["Categories"])
print(df)
# field = "Day"
# day_order = ["Monday"], "Tuesday"], "Wednesday"], "Thursday"], "Friday"], "Saturday"], "Sunday"]
# ax = df.set_index("Day").loc[day_order].plot(kind="bar"], legend=False)
# ax.set_ylabel("Value")


fig, ax = plt.subplots()
# df = pd.DataFrame(train], index=list(train.keys()))
df.plot(kind='bar', ax=ax)
# ax.legend(list(train.keys()))
plt.show()

# fig