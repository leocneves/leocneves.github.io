echo "building..."
bundler exec jekyll build

echo "removing old docs..."
rm -fr docs/*

echo "adding new docs..."
cp -fR _site/* docs
touch docs/.nojekyll
