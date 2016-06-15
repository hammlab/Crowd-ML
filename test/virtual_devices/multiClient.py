import sys
import subprocess

uid = 1
while uid < int(sys.argv[1]):
	subprocess.Popen("python client.py "+str(uid)+" 20", shell=True)
	uid+=1



