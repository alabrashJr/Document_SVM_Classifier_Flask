#!/usr/bin/env python3
import argparse
import json 
import requests 
import pandas as pd 

def get_args():
    '''
    This function parses and return arguments passed in
    '''
    parser = argparse.ArgumentParser(prog='count_gc.py',
                                     description='Compute GC percent of sequence in fasta file')
    parser.add_argument('fasta', help="path to fasta input file")
    args = parser.parse_args()
                                     
    infile = args.fasta
                                     
    return(infile)

def request(id,text):
    r = requests.post(url = "http://localhost:5000/queries", data = {'identifier':id,'text':text},json={"Content-Type":"application/json"}) 
    return r.text 

def get_basename(filename):
    dotsplit = filename.split(".")
    if len(dotsplit) == 1 :
        basename = filename
    else:
        basename = ".".join(dotsplit[:-1])
    return(basename)

if __name__ == "__main__":
    INFILE = get_args()
    OUTFILE = get_basename(INFILE)+".classifier.json"
    
    with open(OUTFILE, "w") as fw:
        with open (INFILE,'r') as fr:
            input=fr.read()
            rtext = request(id=get_basename(INFILE),text=input)
        rtext=json.loads(rtext)
            #strOutput=" ".join([str(cityDic) for cityDic in placesList]).replace('\'',"").replace("{","\n{").lstrip("\n")
            #fw.write(INFILE+"\n")
        json.dump(output ,fw , indent=4)
    

    