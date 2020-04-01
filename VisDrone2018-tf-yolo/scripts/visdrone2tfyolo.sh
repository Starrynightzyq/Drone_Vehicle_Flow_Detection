SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

rm -rf $SHELL_FOLDER/../train.txt
rm -rf $SHELL_FOLDER/../test.txt

cd $SHELL_FOLDER
python3 $SHELL_FOLDER/visdrone2tfyolo.py
