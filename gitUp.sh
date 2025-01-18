git pull origin main --allow-unrelated-histories

git branch -M main
git push -u origin main


git add .
git commit -m "update: $(date '+%d-%m-$Y %H:%M:%S')"
git push

