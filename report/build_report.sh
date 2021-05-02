#!/bin/sh

image_result_dir=$1
N=200
random_source=$(echo random_source)

[ -z $image_result_dir ] && exit -1;

echo "\\title{All predictions on validation data for various super resolution models}"
echo "\\author{Michal Parusinski}"
echo "\\date{\\today}"

echo "\\documentclass[12pt]{article}"
echo "\\usepackage{longtable,tabu}"
echo "\\usepackage{graphicx}"
echo "\\usepackage[margin=0.5in]{geometry}"

echo "\\begin{document}"

echo "\\maketitle"

echo "\\section{Predictions}"

echo "\\begin{longtabu} to \\textwidth {"
echo "        |X[2, c]"
echo "        |X[2, c]"
echo "        |X[2, c]"
echo "        |X[2, c]"
echo "        |X[2, c]"
echo "        |X[2, c]"
echo "        |X[2, c]|"
echo "    }"
echo "    \\hline"
echo "    64 x 64 (ground truth) & Upsampled 32 x 32 & Pre upsampling & Post upsampling & Srcnn & F-Srcnn & ddbpn \\\\"
echo "    \\hline"
echo "    \\endhead"

for fn in `ls $image_result_dir/ground_truth/ | sort -R --random-source=$random_source | tail -$N`; do
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/ground_truth/$fn} & "
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/augmented/upsample_nearest/$fn} & "
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/augmented/pre_upsampling/$fn} & "
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/augmented/post_upsampling/$fn} & "
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/augmented/srcnn/$fn} & "
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/augmented/fsrcnn/$fn} & "
    echo "  \includegraphics[width=\linewidth]{$image_result_dir/augmented/d_dbpn/$fn} \\\\ "
done

echo "\\end{longtabu}"

echo "\\end{document}"
