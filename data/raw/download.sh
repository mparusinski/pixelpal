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
