# -*- coding: utf-8 -*-
import subprocess
import os
import time

subprocess.call('rm result.csv', shell=True)
subprocess.call(" rm elf_image/*", shell=True)
subprocess.call("python3 ELF_CNN/make_image.py", shell=True)
subprocess.call("python3 ELF_CNN/test_ver0.3.py", shell=True)

