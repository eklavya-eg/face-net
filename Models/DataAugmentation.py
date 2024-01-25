import cv2
import os

class DataAugmentation:
    def __init__(self, in_path):
        with open("Data/dataset.csv", "w") as file:
            file.write("Anchor,Positive,Negative\n")
            for ldir in os.listdir(in_path):
                total = 0
                if "72" in ldir : name = "72"
                elif "5" in ldir : name = "5"
                for individuals in os.listdir(os.path.join(in_path, ldir)):
                    image = os.listdir(os.path.join(in_path, ldir, individuals))[0]
                    image = cv2.imread(os.path.join(in_path, ldir, individuals, image))
                    image = cv2.flip(image, 1)
                    cv2.imwrite(os.path.join(in_path, ldir, individuals, f"{name}.png"), image)




if __name__ == "__main__":
    path = "M:/Datasets/FaceRecognition"
    DataAugmentation(path)