import os

def goodRenamer():
    i = 0 
    path = "//Users//nitinabrol//Downloads//goodRenamerFiles//img//"
    leng = len(os.listdir(path)) - 1
    for filename in os.listdir(path):
        names = f"Image - {i} of {leng}.jpg"
        source = path + filename
        destination = path + names
        
        os.rename(source,destination)
        i = i + 1
    
    
if __name__ == "__main__":
    goodRenamer()
