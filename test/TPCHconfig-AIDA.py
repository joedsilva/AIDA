import os
from aida.aida import *

host = 'server-v2'
dbname = 'sf01'
user = 'sf01'
passwd = 'sf01'
jobName = 'test'
port = 55660

SF = 0.01 #used by query 11. indicate the scale factor of the tpch database.

udfVSvtable = False

outputDir = 'output'

def thisJobName(filename):
    return os.path.basename(filename);

def getDBC(jobName):
    return AIDA.connect(host, dbname, user, passwd, jobName, port);
