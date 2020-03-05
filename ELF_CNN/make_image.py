# -*- coding: utf-8 -*-
import os
import imagehash
import numpy
import math
import glob
from PIL import Image, ImageDraw
import re
DATAPATH = "test"
#PATHS = set(os.listdir(DATAPATH))
N = 64
'''
#지정 경로의 파일을 이미지화 시키고
for relpath in PATHS :
    print(relpath)
    PATH = DATAPATH + relpath                                                   ## 이미지 하위 디렉터리 지정
    #print(PATH)
'''
    # 디렉토리 파일목록 불러오기
for FILENAME in os.listdir(DATAPATH):                                    ## 파일 하나씩 지정
    print(FILENAME)
    Bytes = numpy.fromfile('test/'+FILENAME,dtype='uint8')
    ## 파일의 바이너리 데이터로부터 배열을 구축, 데이터 타입은 부호 없는 8비트 정수
    if len(Bytes) >= N * N:                                                 ## 파일 크기가 (64*64)보다 큰 경우
        Compression = math.ceil(len(Bytes) / (N * N))                       ## (64*64)로 나눠준 후 소수점에서 올림
        ConvertTypeArray = numpy.zeros(N * N)                               ## (실수) 0으로 초기화 된 (64*64)의 1차원 배열 생성
        CompBytes = numpy.zeros(N * N, dtype='uint8')                       ## (정수) 0으로 초기화 된 (64*64)의 1차원 배열 생성

        ## Compression Bytes : average compression
                                                                                ## 파일 축소 작업  ex) 파일 크기가 4배면 4바이트를 더한 다음 Compression으로 나눠줌
                                                                                ## ex) 1~4바이트 -> 1바이트 축소
        for i in range(len(Bytes)):
            index = int(i / Compression)                                    ##
            ConvertTypeArray[index] = ConvertTypeArray[index] + Bytes[i]    ##

        for i in range(len(CompBytes)):
            CompBytes[i] = int(ConvertTypeArray[i] / Compression)

        Bytes = CompBytes                                                   ## 축소된 값 입력

    else:                                                                   ## 파일 크기 (64*64)보다 작은 경우
        Multi = (N * N) / len(Bytes)
        tmp=list()
        for i in range(len(Bytes)):
            tmp.append(0)

        for i in range(int(Multi)):
            Bytes = numpy.append(Bytes, tmp)
            
    im = Image.new('RGB', (N, N)) #3x8픽셀의 트루컬러이고 N의 크기를 가졋고 색상도 N

    
    im.save('elf_image/' + FILENAME + '.png')

    im = Image.open('elf_image/' + FILENAME + '.png')
        

    draw = ImageDraw.Draw(im)
    DrawCount = 0

    

    for i in range(N):

        for j in range(N):
            draw.line((j, i) + (j + 1, i), fill=(Bytes[DrawCount], Bytes[DrawCount], Bytes[DrawCount]))
            DrawCount = DrawCount + 1

            if DrawCount >= len(Bytes):
                break

        if DrawCount >= len(Bytes):
            break

    del draw

    # write to stdout
        #os.remove(FILENAME +'.png')
        #f_name = FILENAME.split('\\')
        #f_name = f_name.pop()
        #print(f_name)
    im.save('elf_image/' + FILENAME + '.png')
print('image saved!')
