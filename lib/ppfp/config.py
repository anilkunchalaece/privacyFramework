import json

def getConfig():
    with open("config.json","r") as fd :
        return json.load(fd)