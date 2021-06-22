#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PyInstaller.__main__ import run
from gui.image import image_rc

# -F:打包成一个EXE文件
# -w:不带console输出控制台，window窗体格式
# --paths：依赖包路径
# --icon：图标
# --noupx：不用upx压缩
# --clean：清理掉临时文件


if __name__ == '__main__':
    opts = ['-w', '--icon=image/bankCard.ico', '--clean', 'bankCardIdentify.py', '--hidden-import=queue']  # 不带控制台
    #opts = ['--icon=image/bankCard.ico', '--clean', 'bankCardIdentify.py', '--hidden-import=queue'] # 带控制台
    run(opts)
