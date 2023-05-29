rm -rf ./dependencies/igibson-dev/build


# if [ "$(uname)" == "Darwin" ]; then
# fi
pip install --upgrade-strategy only-if-needed -e ./dependencies/igibson-dev/
pip install --upgrade-strategy only-if-needed -e .
