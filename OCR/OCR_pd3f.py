import time
import requests
import os
import sys

os.chdir(r"C:\Users\Goegg\Downloads\Backup PDFs\done_1951-1999")

def OCRme(filename):
    files = {'pdf': (filename, open('.\\%s' % filename, 'rb'))}
    response = requests.post('http://localhost:1616', files=files, data={'lang': 'de'})
    id = response.json()['id']
    
    while True:
        r = requests.get(f"http://localhost:1616/update/{id}")
        j = r.json()
        if 'text' in j:
            break
        time.sleep(1)
    print("Down with %s" % filename)
    return(j['text'])

g=open("C:\\Users\\Goegg\\OneDrive\\Dokumente\\Uni\\Master\\Masterarbeit\\Daten\\Presseinfos\\Errors.txt", "w", encoding="utf-8")
for filename in os.listdir(os.getcwd()):
    try:
        f=open(os.path.join(r"C:\Users\Goegg\Downloads\Backup PDFs\pd3f_test", os.path.splitext(filename)[0]+".txt"),'w', encoding="utf-8")
        f.write(OCRme(filename))
        f.close()
    except:
        e = sys.exc_info()[0]
        g.write("Affected file: %s.\n   The Error Message is: %s" % (str(filename), str(e)))
        continue
g.close()
    
