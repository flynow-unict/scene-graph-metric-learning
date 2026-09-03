import os
import paramiko
from dotenv import load_dotenv

load_dotenv("cloud/.env")
CLUSTER_USER = os.getenv("CLUSTER_USER")
CLUSTER_HOST = os.getenv("CLUSTER_HOST")
CLUSTER_PW = os.getenv("CLUSTER_PW")

def run_ssh_command(command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=CLUSTER_HOST, username=CLUSTER_USER, password=CLUSTER_PW)
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())
    ssh.close()

run_ssh_command("ls -la /home/lsisnt03h03c351k/Progetto/logs/ | tail -n 5")
run_ssh_command("cat $(ls -t /home/lsisnt03h03c351k/Progetto/logs/*.out | head -n 1)")
