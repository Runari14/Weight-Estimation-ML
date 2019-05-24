import hashlib

import hashlib
"""
To be used to verify if stuff is finished upploading to the cair gpu's (no progressbar

usage:
    1. on your local computer set the path to whatever you want to upload.
    2. when finished replace compare_against with the returned string
    3. upload to cair this file with new hash (and whatever you want to upload)
    4. run this program on the the server, and retry untill it says "they are equal"
    5. you are good to go! :D 
"""


# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
compare_against = "040a8dc162cf52bf5ac84c0850519608b005bc1e4bdcfad6fcbb1574b444d466"

sha3 = hashlib.sha3_256()

with open("Dataset/food-101.tar.gz", 'rb') as f:
    while True:
        data = f.read(BUF_SIZE)
        if not data:
            break
        sha3.update(data)


print("SHA3: {0}".format(sha3.hexdigest()))

if not compare_against == "":
    if compare_against == sha3.hexdigest():
        print("they are equal")
    else:
        print("they are unequal")
