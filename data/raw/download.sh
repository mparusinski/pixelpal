#!/bin/bash

read -p "Delete files? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -rf '*.zip' '*.tar.gz'
fi

# TESTING
curl -o clearlooks.zip "https://codeload.github.com/jpfleury/clearlooks-phenix/zip/master"
curl -o tango.tar.gz "https://dllb2.pling.com/api/files/download/j/eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6IjE0OTY4Njk3MjkiLCJ1IjpudWxsLCJsdCI6ImRvd25sb2FkIiwicyI6IjNkN2U1MzEzYmE4ZDFhZDRhZmVhZDgzZjlkNzUxYTI0Zjk5ODM3NzcyY2VhMDU1MDBkMjUyMTE2YmJkZjI2N2QxYjlkMDc5ZjgzNTVjZGRmODFhMDM5M2U0MGMxNTU0NGFmYjc4NjEwZjE3NTBhZTcxZjBiNzIyOWQxMDFiNDJiIiwidCI6MTU4MTc2NTA1Niwic3RmcCI6IjViZGViOTZmY2E5MTFiMDYzMzQxMWM1OTRhYmM2YjA5Iiwic3RpcCI6IjJhMDE6ZTM1OjJlNzQ6MmI3MDo1OTJmOmNmMTpjMTQzOmI0MzIifQ.Zv0zqhHwgTrokcSIWqAd6X676hoEVl4eH5pQhQxs5dY/Tango2-2017.1.tar.gz"

# TRAINING
curl -o adwaita.tar.gz https://gitlab.gnome.org/GNOME/adwaita-icon-theme/-/archive/master/adwaita-icon-theme-master.tar.gz
curl -o oxygen.zip https://codeload.github.com/KDE/oxygen-icons/zip/master
curl -o arc.zip https://codeload.github.com/horst3180/arc-theme/zip/master
curl -o equilux.zip https://codeload.github.com/ddnexus/equilux-theme/zip/equilux-dev
curl -o faba.zip https://codeload.github.com/snwh/faba-icon-theme/zip/master
curl -o moka.zip https://codeload.github.com/CSRedRat/moka-gtk-theme/zip/master
curl -o zayronxio.zip https://codeload.github.com/zayronxio/Zafiro-icons/zip/master
curl -o wildfire.zip https://codeload.github.com/xenlism/wildfire/zip/master
