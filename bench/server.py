import os, signal, subprocess, time, urllib.request, urllib.error

def wait_ready(proc, prort, timeout=600):
    debut=