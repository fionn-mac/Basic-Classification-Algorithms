import os
import random
classes = ['galsworthy/','galsworthy_2/','mill/','shelley/','thackerey/','thackerey_2/','wordsmith_prose/','cia/','johnfranklinjameson/','diplomaticcorr/']

for c in classes:

    listing = os.listdir(c)
    for filename in listing:

        r=random.random()
        if r<=0.2:
            pass
            os.system("mv "+c+filename+" ../test/"+c+filename)
