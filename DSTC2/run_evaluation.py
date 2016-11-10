import logging, os, subprocess
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

featured_metrics = []
all_metrics = []

def processFolder(folder) :
    # check for each entry, there is a .test.json, .dev.score.csv
    entries = set()
    for file_name in os.listdir(folder) :
        if file_name.startswith("entry") :
            entries.add(file_name[:6])
    if len(entries) > 5 :
        logger.error("%s has too many entries", folder)
    _entries = ["entry%i"%i for i in range(len(entries))]
    if ",".join(sorted(_entries)) != ",".join(sorted(entries)):
        logger.error("For %s expected entries %s, got %s", folder, str(_entries), str(sorted(list(entries))))
        
    for entry in entries :
        required_files = [os.path.join(folder, entry+f) for f in (".test.json", ".dev.score.csv", ".txt")]
        for required_file in required_files :
            if not os.path.exists(required_file) :
                logger.error("%s does not exist", required_file)
    # run checkTrack on .json
    with open(os.devnull, "w") as fnull:
        for entry in entries :
            check_file_name = os.path.join(folder, entry+".check.txt")
            if os.path.exists(check_file_name) :
                continue
            with open(check_file_name,"w") as check_file :
                subprocess.call(["python", "../scripts/check_track.py", "--dataset",
                      "dstc2_test","--dataroot","../data","--ontology","../scripts/config/ontology_dstc2.json",
                      "--trackfile",os.path.join(folder, entry+ ".test.json")]
                    ,stdout = check_file, stderr = fnull)
            with open(check_file_name, "r") as check_file: 
                for line in check_file :
                    if "no errors" not in line :
                        logger.warning("check_track error for %s/%s: "+line.strip(), folder, entry)
    
    for entry in entries :
        score_file_name = os.path.join(folder, entry+".test.score.csv")
        if os.path.exists(score_file_name):
            continue
        with open(os.devnull, "w") as fnull:
            logger.info("Scoring %s/%s", folder, entry)
            subprocess.call(["python", "../scripts/score.py", "--dataset","dstc2_test",
                            "--dataroot","../data","--trackfile",os.path.join(folder, entry+".test.json"),
                            "--scorefile",score_file_name,"--ontology","../scripts/config/ontology_dstc2.json"],
               # stdout = fnull, stderr = fnull
                )
    # add featured metrics:
    for entry in entries :
        score_file_name = os.path.join(folder, entry+".test.score.csv")
        for line_num, line in enumerate(open(score_file_name)) :
            if line_num == 0:
                continue
            fields = tuple([x.strip() for x in line.split(",")[:4]])
            if fields in [
                    ("goal.joint","acc","2","a"),
                    ("goal.joint","l2","2","a"),
                    ("goal.joint","roc.v2_ca05","2","a"),
                    ("method","acc","2","a"),
                    ("method","l2","2","a"),
                    ("method","roc.v2_ca05","2","a"),
                    ("requested.all","acc","2","a"),
                    ("requested.all","l2","2","a"),
                    ("requested.all","roc.v2_ca05","2","a"),
                ] :
                featured_metrics.append(
                        ("%s, %s, "%(folder, entry))+line.strip()
                    )
            all_metrics.append(
                ("%s, %s, "%(folder, entry))+line.strip()
            )
        

def createFeaturedTable():
    header = "team, entry, state_component, stat, schedule, label_scheme, N, result"
    def _sort_key(x) :
        return "".join((x.split(",")[2:6]))+" "+"".join((x.split(",")[:2]))
    featured_metrics.sort(key =_sort_key)
    featured_table = open("featured.csv", "w")
    for line in [header]+featured_metrics:
        featured_table.write(line+"\n")
    # also create all metrics csv
    all_metrics.sort(key =_sort_key)
    all_file = open("all.csv", "w")
    for line in [header] + all_metrics:
        all_file.write(line+"\n")
    

    

teams = []
for file_name in os.listdir(".") :
    if file_name.startswith("team") :
        teams.append(file_name)

for team in teams :
    processFolder(team)
createFeaturedTable()
logger.info("Finished")