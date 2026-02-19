rm -f a1.zip 
pushd submission; zip -r ../a1.zip . --exclude "*__pycache__*"; popd