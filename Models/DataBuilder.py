import os
import random

class DataBuilder:
    def __init__(self, in_path, labels_72):
        try:
            os.mkdir(os.path.join(os.getcwd(), 'Models', "Data"))
        except FileExistsError:
            pass
        with open(os.path.join(os.getcwd(), 'Models', "Data", "Dataset.csv"), "w") as file:
            file.write("Anchor,Positive,Negative\n")
            total = 0
            for ldir in os.listdir(in_path)[:labels_72]:
                negative_index = 0
                nidentities = os.listdir(in_path)
                nidentities.remove(ldir)
                negative_list = []
                maximgs = 36*71+50
                for n in nidentities:
                    if len(negative_list)>maximgs:
                        break
                    lambdafn = lambda x: negative_list.append(os.path.join(in_path, n, x))
                    nidimg = os.listdir(os.path.join(in_path, n))[:72]
                    list(map(lambdafn, nidimg))

                images = os.listdir(os.path.join(in_path, ldir))
                for indivs in range(0, len(images)-1):
                    for indiv in range(indivs+1, len(images)):
                        file.write(",".join([os.path.join(in_path, ldir, images[indivs]), os.path.join(in_path, ldir, images[indiv]), negative_list[negative_index]])+"\n")
                        negative_index+=1
                        total += 1
        print(total)

                        


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "Models", "Datasets", "FaceRecognition", "labels-72")
    DataBuilder(path, 200)