#!/bin/bash

read -p "Delete files? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf '*.zip' '*.tar.gz'
fi

# TESTING
curl -o clearlooks.zip "https://codeload.github.com/jpfleury/clearlooks-phenix/zip/master"
curl -o tango.zip "https://codeload.github.com/Distrotech/tango-icon-theme/zip/master"

# TRAINING
curl -o adwaita.tar.gz https://gitlab.gnome.org/GNOME/adwaita-icon-theme/-/archive/master/adwaita-icon-theme-master.tar.gz
curl -o oxygen.zip https://codeload.github.com/KDE/oxygen-icons/zip/master
curl -o arc.zip https://codeload.github.com/horst3180/arc-theme/zip/master
curl -o equilux.zip https://codeload.github.com/ddnexus/equilux-theme/zip/equilux-dev
curl -o faba.zip https://codeload.github.com/snwh/faba-icon-theme/zip/master
curl -o moka.zip https://codeload.github.com/CSRedRat/moka-gtk-theme/zip/master
curl -o zayronxio.zip https://codeload.github.com/zayronxio/Zafiro-icons/zip/master
curl -o wildfire.zip https://codeload.github.com/xenlism/wildfire/zip/master

# FINAL TEST
curl -o CDE.tar.gz https://dllb2.pling.com/api/files/download/j/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6IjE0NjY2MTE3OTIiLCJ1IjpudWxsLCJsdCI6ImRvd25sb2FkIiwicyI6ImRiMGE4Nzg1NzY0ZjNkYjg2Yjc0Y2NiYTRmYTY0OTViYjA1OWQwNzRmY2MzMjczMzMyMzkyYTYyM2M1NmQ1MjBmNmQ4NTRhOWM5Mjc0MTI5MDIyMWFlNmNiNTM4MWUzNDcxZTY0MTI5NDM5Njg1NjViYTEyYzU5MDA4NzJkY2JkIiwidCI6MTU4MzY2NTQ3OSwic3RmcCI6IjFkMDNjNmQyOTc2NjM2NDEyZTFhMWI4MDM3MTFlY2JkIiwic3RpcCI6IjJhMDE6ZTM1OjJlNzQ6MmI3MDo5ZDM1OjNiMjY6NTliODpmNDRjIn0.iB5fMS05ssG2ZlnwUuo3vLf_EdxgObatdWTAt7ikgKg/36104-CDE.tar.gz
curl -o OS2.zip www.altsan.org/programming/os2/itheme_toolkit_22.zip