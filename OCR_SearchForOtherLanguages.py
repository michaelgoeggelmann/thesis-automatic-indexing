import re
import shutil
import os
FOLDER_PATH = r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\E _OPENTEXT_OTCS_temp_multifile_67053-2736-886564670"

lettersearch = re.compile(r"")

for filename in os.listdir(FOLDER_PATH):
    #print(filename)
    if os.path.splitext(filename)[1] == ".pdf":
        print(os.path.splitext(filename)[1])
        #shutil.move(os.path.join(FOLDER_PATH, filename), r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\Andere Sprachen\Sortieren")
