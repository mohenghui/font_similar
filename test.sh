python lpips_2imgs.py -p0 imgs/0/86.png -p1 imgs/0/115.png --use_gpu
python lpips_2dirs.py -d0 imgs/ex_dir0 -d1 imgs/ex_dir1 -o imgs/example_dists.txt --use_gpu
python lpips_1dir_allpairs.py -d imgs/ex_dir_pair -o imgs/example_dists_pair.txt --use_gpu
python lpips_2dirs.py -d0 imgs/AaJueXingHei70J -d1 imgs/FZFengBPYTJW -o imgs/example_dists.txt --use_gpu