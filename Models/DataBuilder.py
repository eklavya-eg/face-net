import os
import random

class DataBuilder:
    def __init__(self, in_path, labels_72, labels_5):
        try:
            os.mkdir("W:/Projects/FaceRecognition/Models/Data")
        except FileExistsError:
            pass
        with open("W:/Projects/FaceRecognition/Models/Data/Dataset.csv", "w") as file:
            file.write("Anchor,Positive,Negative\n")
            for ldir in os.listdir(in_path):
                total = 0
                if "72" in ldir : limit = labels_72
                elif "5" in ldir : limit = labels_5
                for individuals in os.listdir(os.path.join(in_path, ldir)):
                    if total<limit:
                        labels = os.listdir(os.path.join(in_path, ldir))
                        labels = list(set(labels)-{individuals})
                        random.shuffle(labels)
                        images = os.listdir(os.path.join(in_path, ldir, individuals))
                        anchor = images[0]
                        images = list(set(images)-{anchor})
                        anchor = "/".join([in_path, ldir, individuals, anchor])
                        for i in range(len(images)):
                            pos = "/".join([in_path, ldir, individuals, images[i]])
                            if i != 0:
                                label = os.listdir(os.path.join(in_path, ldir, labels[i]))
                                random.shuffle(label)
                                label = label[0]
                                neg = os.path.join(in_path, ldir, labels[i], label)
                            else:
                                label = os.listdir(os.path.join(in_path, ldir, labels[i]))[0]
                                neg = "/".join([in_path, ldir, labels[i], label])
                            file.write(f"{anchor},{pos},{neg}\n")
                        total += 1


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "Models", "Datasets", "FaceRecognition").split("\\")
    path = "/".join(path)
    DataBuilder(path, 800, 1000)