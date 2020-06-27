#!/bin/bash
wget --no-check-certificate -r "https://docs.google.com/uc?export=download&id=$1" -O $(curl -s "https://drive.google.com/file/d/$1/view?usp=sharing" | grep -o '<title>.*</title>' | cut -d'>' -f2 | awk -F ' - Goo' '{print $1}')
